#include "tensorrelay_stage_executor.h"

#include "gguf.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct tr_stage_metadata {
    bool        stage_marker = false;
    uint32_t    stage_index = 0;
    uint32_t    stage_count = 0;
    uint32_t    first_layer = 0;
    uint32_t    last_layer_exclusive = 0;
    int64_t     tensor_count = 0;
    std::string architecture;
};

struct tr_stage_executor_state {
    std::string last_error;
    std::string shard_path;
    std::string selected_devices;
    std::string architecture;
    int32_t     stage_index = -1;
    int32_t     split_count = 0;
    uint32_t    first_layer = 0;
    uint32_t    last_layer_exclusive = 0;
    uint32_t    slot_count = 0;
    uint32_t    context_length = 0;
    bool        loaded = false;
    std::vector<uint64_t> slot_epochs;
    std::mutex  mutex;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
};

static std::once_flag backend_init_once;

static tr_stage_executor_state * as_state(void * handle) {
    return static_cast<tr_stage_executor_state *>(handle);
}

static int32_t set_error(tr_stage_executor_state * state, const std::string & message, int32_t code) {
    if (state) {
        state->last_error = message;
    }
    return code;
}

static std::string bytes_to_string(const uint8_t * ptr, size_t len) {
    if (ptr == nullptr || len == 0) {
        return {};
    }
    return std::string(reinterpret_cast<const char *>(ptr), len);
}

static bool gguf_bool(const gguf_context * ctx, const char * key, bool & out, std::string & error) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        error = std::string("missing GGUF metadata key: ") + key;
        return false;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_BOOL) {
        error = std::string("GGUF metadata key has wrong type: ") + key;
        return false;
    }
    out = gguf_get_val_bool(ctx, id);
    return true;
}

static bool gguf_u32(const gguf_context * ctx, const char * key, uint32_t & out, std::string & error) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        error = std::string("missing GGUF metadata key: ") + key;
        return false;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        error = std::string("GGUF metadata key has wrong type: ") + key;
        return false;
    }
    out = gguf_get_val_u32(ctx, id);
    return true;
}

static bool gguf_string(const gguf_context * ctx, const char * key, std::string & out, std::string & error) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        error = std::string("missing GGUF metadata key: ") + key;
        return false;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        error = std::string("GGUF metadata key has wrong type: ") + key;
        return false;
    }
    const char * value = gguf_get_val_str(ctx, id);
    out = value == nullptr ? std::string() : std::string(value);
    return true;
}

static bool is_supported_architecture(const std::string & architecture) {
    static const std::set<std::string> supported = {
        "qwen2",
        "qwen2moe",
        "qwen3",
        "qwen3moe",
        "qwen35",
        "qwen35moe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3n",
        "gemma4",
        "gemma4-assistant",
    };
    return supported.find(architecture) != supported.end();
}

static bool read_stage_metadata(const std::string & shard_path, tr_stage_metadata & out, std::string & error) {
    gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
        gguf_init_from_file(shard_path.c_str(), params), gguf_free);
    if (!ctx) {
        error = "failed to read GGUF metadata from stage shard: " + shard_path;
        return false;
    }
    if (!gguf_bool(ctx.get(), "tensorrelay.stage", out.stage_marker, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.index", out.stage_index, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.count", out.stage_count, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.first_layer", out.first_layer, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.last_layer_exclusive", out.last_layer_exclusive, error) ||
        !gguf_string(ctx.get(), "general.architecture", out.architecture, error)) {
        return false;
    }
    out.tensor_count = gguf_get_n_tensors(ctx.get());
    return true;
}

static bool validate_stage_metadata(
        const tr_stage_metadata & meta,
        const tr_stage_executor_load_params * params,
        std::string & error) {
    if (!meta.stage_marker) {
        error = "GGUF artifact is not marked as a TensorRelay layer-stage shard";
        return false;
    }
    if (meta.stage_index != static_cast<uint32_t>(params->stage_index)) {
        error = "GGUF stage index does not match assigned runtime stage";
        return false;
    }
    if (meta.stage_count != static_cast<uint32_t>(params->split_count)) {
        error = "GGUF stage count does not match assigned split count";
        return false;
    }
    if (meta.first_layer >= meta.last_layer_exclusive) {
        error = "GGUF stage layer range is empty or invalid";
        return false;
    }
    if (meta.tensor_count <= 0) {
        error = "GGUF stage shard contains no tensors";
        return false;
    }
    if (!is_supported_architecture(meta.architecture)) {
        error = "unsupported TensorRelay stage architecture: " + meta.architecture;
        return false;
    }
    return true;
}

static int request_max_tokens(const json & body) {
    int max_tokens = 128;
    const auto read_int = [&](const char * key, int current) {
        if (!body.contains(key) || !body.at(key).is_number_integer()) {
            return current;
        }
        return body.at(key).get<int>();
    };
    max_tokens = read_int("max_tokens", max_tokens);
    max_tokens = read_int("max_completion_tokens", max_tokens);
    return std::max(1, std::min(max_tokens, 512));
}

static std::string content_to_text(const json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        return {};
    }
    std::string out;
    for (const auto & part : content) {
        if (!part.is_object()) {
            continue;
        }
        if (part.contains("text") && part.at("text").is_string()) {
            if (!out.empty()) {
                out += "\n";
            }
            out += part.at("text").get<std::string>();
        }
    }
    return out;
}

static bool request_to_prompt(
        llama_model * model,
        const std::string & payload,
        std::string & prompt,
        int & max_tokens,
        std::string & error) {
    json body;
    try {
        body = json::parse(payload);
    } catch (const std::exception & e) {
        error = std::string("invalid OpenAI request JSON: ") + e.what();
        return false;
    }

    max_tokens = request_max_tokens(body);

    if (body.contains("prompt") && body.at("prompt").is_string()) {
        prompt = body.at("prompt").get<std::string>();
        return !prompt.empty();
    }

    if (!body.contains("messages") || !body.at("messages").is_array()) {
        error = "OpenAI request JSON must contain messages[] or prompt";
        return false;
    }

    std::vector<std::pair<std::string, std::string>> owned_messages;
    for (const auto & item : body.at("messages")) {
        if (!item.is_object()) {
            continue;
        }
        const std::string role = item.contains("role") && item.at("role").is_string()
            ? item.at("role").get<std::string>()
            : "user";
        const std::string content = item.contains("content") ? content_to_text(item.at("content")) : std::string();
        if (!content.empty()) {
            owned_messages.emplace_back(role, content);
        }
    }
    if (owned_messages.empty()) {
        error = "OpenAI request JSON messages[] did not contain text content";
        return false;
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(owned_messages.size());
    for (const auto & item : owned_messages) {
        chat.push_back({ item.first.c_str(), item.second.c_str() });
    }

    const char * tmpl = llama_model_chat_template(model, nullptr);
    int formatted_len = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, nullptr, 0);
    if (formatted_len < 0) {
        error = "failed to apply model chat template";
        return false;
    }
    std::vector<char> formatted(static_cast<size_t>(formatted_len) + 1);
    formatted_len = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, formatted.data(), formatted.size());
    if (formatted_len < 0) {
        error = "failed to apply model chat template";
        return false;
    }
    prompt.assign(formatted.data(), static_cast<size_t>(formatted_len));
    return !prompt.empty();
}

static bool token_to_piece(const llama_vocab * vocab, llama_token token, std::string & piece, std::string & error) {
    char stack_buf[256];
    int n = llama_token_to_piece(vocab, token, stack_buf, sizeof(stack_buf), 0, true);
    if (n >= 0) {
        piece.assign(stack_buf, static_cast<size_t>(n));
        return true;
    }
    std::vector<char> buf(static_cast<size_t>(-n) + 1);
    n = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
    if (n < 0) {
        error = "failed to convert sampled token to text";
        return false;
    }
    piece.assign(buf.data(), static_cast<size_t>(n));
    return true;
}

struct token_batch_storage {
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<int32_t> n_seq_ids;
    std::vector<llama_seq_id> seq_ids;
    std::vector<llama_seq_id *> seq_id_ptrs;
    std::vector<int8_t> logits;
    llama_batch batch{};

    void reset(const std::vector<llama_token> & input_tokens, llama_seq_id seq_id, llama_pos pos_start) {
        tokens = input_tokens;
        positions.resize(tokens.size());
        n_seq_ids.assign(tokens.size(), 1);
        seq_ids.assign(tokens.size(), seq_id);
        seq_id_ptrs.resize(tokens.size());
        logits.assign(tokens.size(), 0);

        for (size_t i = 0; i < tokens.size(); ++i) {
            positions[i] = pos_start + static_cast<llama_pos>(i);
            seq_id_ptrs[i] = &seq_ids[i];
        }
        if (!logits.empty()) {
            logits.back() = 1;
        }

        batch = {};
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        batch.token = tokens.data();
        batch.embd = nullptr;
        batch.pos = positions.data();
        batch.n_seq_id = n_seq_ids.data();
        batch.seq_id = seq_id_ptrs.data();
        batch.logits = logits.data();
    }
};

static bool generate_single_stage(
        tr_stage_executor_state * state,
        uint32_t slot_id,
        const std::string & payload,
        std::string & out,
        std::string & error) {
    if (state->model == nullptr || state->ctx == nullptr || state->vocab == nullptr) {
        error = "single-stage llama runtime is not initialized";
        return false;
    }

    std::string prompt;
    int max_tokens = 0;
    if (!request_to_prompt(state->model, payload, prompt, max_tokens, error)) {
        return false;
    }

    const llama_seq_id seq_id = static_cast<llama_seq_id>(slot_id);
    llama_memory_seq_rm(llama_get_memory(state->ctx), seq_id, -1, -1);

    const int n_prompt = -llama_tokenize(
        state->vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        error = "failed to tokenize prompt";
        return false;
    }
    if (n_prompt + max_tokens > static_cast<int>(llama_n_ctx(state->ctx))) {
        error = "prompt and requested output exceed executor context length";
        return false;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(
            state->vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()),
            true,
            true) < 0) {
        error = "failed to tokenize prompt";
        return false;
    }

    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_guard(sampler, llama_sampler_free);

    token_batch_storage batch;
    batch.reset(prompt_tokens, seq_id, 0);
    llama_token next_token = LLAMA_TOKEN_NULL;
    for (int i = 0; i < max_tokens; ++i) {
        const int decode_rc = llama_decode(state->ctx, batch.batch);
        if (decode_rc != 0) {
            error = "llama_decode failed with code " + std::to_string(decode_rc);
            return false;
        }

        next_token = llama_sampler_sample(sampler, state->ctx, -1);
        if (llama_vocab_is_eog(state->vocab, next_token)) {
            break;
        }
        std::string piece;
        if (!token_to_piece(state->vocab, next_token, piece, error)) {
            return false;
        }
        out += piece;
        batch.reset({ next_token }, seq_id, static_cast<llama_pos>(prompt_tokens.size() + i));
    }

    return true;
}

extern "C" {

uint32_t tr_stage_executor_abi_version(void) {
    return TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION;
}

int32_t tr_stage_executor_create(void ** out_handle) {
    if (out_handle == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    auto state = std::make_unique<tr_stage_executor_state>();
    state->last_error = "";
    *out_handle = state.release();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

void tr_stage_executor_destroy(void * handle) {
    auto * state = as_state(handle);
    if (state != nullptr) {
        if (state->ctx != nullptr) {
            llama_free(state->ctx);
            state->ctx = nullptr;
        }
        if (state->model != nullptr) {
            llama_model_free(state->model);
            state->model = nullptr;
        }
    }
    delete state;
}

int32_t tr_stage_executor_load(void * handle, const tr_stage_executor_load_params * params) {
    auto * state = as_state(handle);
    if (state == nullptr || params == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    if (params->abi_version != TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION) {
        return set_error(state, "stage executor ABI version mismatch", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (params->stage_index < 0 || params->split_count <= 0 || params->stage_index >= params->split_count) {
        return set_error(state, "invalid stage index or split count", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (params->slot_count == 0) {
        return set_error(state, "slot_count must be greater than zero", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    std::string shard_path = bytes_to_string(params->shard_path_ptr, params->shard_path_len);
    if (shard_path.empty()) {
        return set_error(state, "shard path is empty", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(std::filesystem::path(shard_path), ec)) {
        return set_error(state, "stage shard file does not exist: " + shard_path, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    tr_stage_metadata metadata;
    std::string metadata_error;
    if (!read_stage_metadata(shard_path, metadata, metadata_error)) {
        return set_error(state, metadata_error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (!validate_stage_metadata(metadata, params, metadata_error)) {
        return set_error(state, metadata_error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }

    if (state->ctx != nullptr) {
        llama_free(state->ctx);
        state->ctx = nullptr;
    }
    if (state->model != nullptr) {
        llama_model_free(state->model);
        state->model = nullptr;
    }

    state->shard_path = shard_path;
    state->selected_devices = bytes_to_string(params->selected_devices_ptr, params->selected_devices_len);
    state->architecture = metadata.architecture;
    state->stage_index = params->stage_index;
    state->split_count = params->split_count;
    state->first_layer = metadata.first_layer;
    state->last_layer_exclusive = metadata.last_layer_exclusive;
    state->slot_count = params->slot_count;
    state->context_length = params->context_length;
    state->vocab = nullptr;

    if (params->split_count == 1) {
        std::call_once(backend_init_once, []() {
            llama_backend_init();
            ggml_backend_load_all();
        });
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = state->selected_devices.empty() ? 0 : 999;
        state->model = llama_model_load_from_file(state->shard_path.c_str(), model_params);
        if (state->model == nullptr) {
            return set_error(state, "failed to load single-stage llama model", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        state->vocab = llama_model_get_vocab(state->model);
        if (state->vocab == nullptr) {
            return set_error(state, "single-stage llama model has no vocabulary", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = state->context_length > 0 ? state->context_length : 4096;
        ctx_params.n_batch = std::min<uint32_t>(ctx_params.n_ctx, 512);
        ctx_params.n_seq_max = std::max<uint32_t>(state->slot_count, 1);
        ctx_params.no_perf = true;
        state->ctx = llama_init_from_model(state->model, ctx_params);
        if (state->ctx == nullptr) {
            llama_model_free(state->model);
            state->model = nullptr;
            state->vocab = nullptr;
            return set_error(state, "failed to create single-stage llama context", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
    }

    state->loaded = true;
    state->last_error.clear();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

int32_t tr_stage_executor_alloc_slots(void * handle, uint32_t slot_count) {
    auto * state = as_state(handle);
    if (state == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    if (!state->loaded) {
        return set_error(state, "stage executor is not loaded", TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED);
    }
    if (slot_count == 0 || slot_count != state->slot_count) {
        return set_error(state, "slot allocation count does not match loaded runtime config", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    state->slot_epochs.assign(slot_count, 0);
    state->last_error.clear();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

int32_t tr_stage_executor_execute_batch(void * handle, tr_stage_executor_batch_call * call) {
    auto * state = as_state(handle);
    if (state == nullptr || call == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->loaded || state->slot_epochs.empty()) {
        return set_error(state, "stage executor slots are not allocated", TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED);
    }
    if (call->abi_version != TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION) {
        return set_error(state, "stage executor batch ABI version mismatch", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (call->input_count > call->outputs_len) {
        return set_error(state, "outputs_len must be at least input_count", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    for (uint32_t i = 0; i < call->input_count; ++i) {
        if (call->inputs_ptr == nullptr || call->outputs_ptr == nullptr) {
            return set_error(state, "batch inputs and outputs are required", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        if (call->inputs_ptr[i].slot_id >= state->slot_epochs.size()) {
            return set_error(state, "batch input references an unknown slot", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        const auto & input = call->inputs_ptr[i];
        const uint64_t input_end = static_cast<uint64_t>(input.activation_offset) + static_cast<uint64_t>(input.activation_len);
        if (input_end > call->input_bytes_len) {
            return set_error(state, "batch input byte span is outside input buffer", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
    }
    if (call->output_bytes_ptr == nullptr || call->output_bytes_len == 0) {
        return set_error(state, "batch output byte buffer is required", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (state->split_count != 1) {
        return set_error(
            state,
            "TensorRelay multi-stage graph-boundary execution is not implemented yet",
            TENSORRELAY_STAGE_EXECUTOR_ERR_UNSUPPORTED);
    }

    size_t output_cursor = 0;
    for (uint32_t i = 0; i < call->input_count; ++i) {
        const auto & input = call->inputs_ptr[i];
        auto & output = call->outputs_ptr[i];
        const auto * input_begin = call->input_bytes_ptr + input.activation_offset;
        const std::string input_payload(reinterpret_cast<const char *>(input_begin), input.activation_len);
        std::string payload;
        std::string error;

        if ((input.flags & TENSORRELAY_STAGE_EXECUTOR_INPUT_FLAG_REQUEST_JSON) == 0) {
            return set_error(
                state,
                "single-stage execution requires OpenAI request JSON input",
                TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        if (!generate_single_stage(state, input.slot_id, input_payload, payload, error)) {
            return set_error(state, error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        output.flags = TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_FINAL_TEXT;
        output.sampled_token_id = -1;

        if (output_cursor + payload.size() > call->output_bytes_len) {
            return set_error(state, "batch output buffer is too small", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        std::memcpy(call->output_bytes_ptr + output_cursor, payload.data(), payload.size());
        output.slot_id = input.slot_id;
        output.request_epoch = input.request_epoch;
        output.activation_offset = static_cast<uint32_t>(output_cursor);
        output.activation_len = static_cast<uint32_t>(payload.size());
        output_cursor += payload.size();
        state->slot_epochs[input.slot_id] = std::max(state->slot_epochs[input.slot_id], input.request_epoch);
    }

    state->last_error.clear();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

int32_t tr_stage_executor_release_slot(void * handle, uint32_t slot_id, uint64_t request_epoch) {
    auto * state = as_state(handle);
    if (state == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (slot_id >= state->slot_epochs.size()) {
        return set_error(state, "release references an unknown slot", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (state->ctx != nullptr) {
        llama_memory_seq_rm(llama_get_memory(state->ctx), static_cast<llama_seq_id>(slot_id), -1, -1);
    }
    state->slot_epochs[slot_id] = std::max(state->slot_epochs[slot_id], request_epoch);
    state->last_error.clear();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

size_t tr_stage_executor_last_error(void * handle, uint8_t * out_ptr, size_t out_len) {
    auto * state = as_state(handle);
    const std::string message = state == nullptr ? "no executor handle" : state->last_error;
    if (out_ptr == nullptr || out_len == 0) {
        return message.size();
    }
    const size_t n = std::min(out_len, message.size());
    std::memcpy(out_ptr, message.data(), n);
    return n;
}

}
