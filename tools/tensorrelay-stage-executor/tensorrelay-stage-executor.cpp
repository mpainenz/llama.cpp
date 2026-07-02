#include "tensorrelay_stage_executor.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct tr_stage_metadata {
    bool        has_tensorrelay_stage_metadata = false;
    bool        has_legacy_split_metadata = false;
    bool        has_architecture = false;
    bool        stage_marker = false;
    uint32_t    stage_index = 0;
    uint32_t    stage_count = 0;
    uint32_t    first_layer = 0;
    uint32_t    last_layer_exclusive = 0;
    uint32_t    hidden_dim = 0;
    uint32_t    legacy_split_no = 0;
    uint32_t    legacy_split_count = 0;
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
    uint32_t    hidden_dim = 0;
    uint32_t    slot_count = 0;
    uint32_t    context_length = 0;
    bool        loaded = false;
    std::vector<std::string> device_names;
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

static bool read_gguf_u32_value(const gguf_context * ctx, int64_t id, uint32_t & out) {
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:
            out = gguf_get_val_u8(ctx, id);
            return true;
        case GGUF_TYPE_UINT16:
            out = gguf_get_val_u16(ctx, id);
            return true;
        case GGUF_TYPE_UINT32:
            out = gguf_get_val_u32(ctx, id);
            return true;
        case GGUF_TYPE_UINT64: {
            const uint64_t value = gguf_get_val_u64(ctx, id);
            if (value > std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            out = static_cast<uint32_t>(value);
            return true;
        }
        case GGUF_TYPE_INT8: {
            const int8_t value = gguf_get_val_i8(ctx, id);
            if (value < 0) {
                return false;
            }
            out = static_cast<uint32_t>(value);
            return true;
        }
        case GGUF_TYPE_INT16: {
            const int16_t value = gguf_get_val_i16(ctx, id);
            if (value < 0) {
                return false;
            }
            out = static_cast<uint32_t>(value);
            return true;
        }
        case GGUF_TYPE_INT32: {
            const int32_t value = gguf_get_val_i32(ctx, id);
            if (value < 0) {
                return false;
            }
            out = static_cast<uint32_t>(value);
            return true;
        }
        case GGUF_TYPE_INT64: {
            const int64_t value = gguf_get_val_i64(ctx, id);
            if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            out = static_cast<uint32_t>(value);
            return true;
        }
        default:
            return false;
    }
}

static bool gguf_u32(const gguf_context * ctx, const char * key, uint32_t & out, std::string & error) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        error = std::string("missing GGUF metadata key: ") + key;
        return false;
    }
    if (!read_gguf_u32_value(ctx, id, out)) {
        error = std::string("GGUF metadata key has wrong type: ") + key;
        return false;
    }
    return true;
}

static bool gguf_optional_u32(const gguf_context * ctx, const char * key, uint32_t & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return false;
    }
    return read_gguf_u32_value(ctx, id, out);
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

static bool is_split_graph_execution_architecture(const std::string & architecture) {
    static const std::set<std::string> supported = {
        "qwen2",
        "qwen2moe",
        "qwen3",
        "qwen3moe",
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
    out.has_architecture = gguf_find_key(ctx.get(), "general.architecture") >= 0;
    if (out.has_architecture && !gguf_string(ctx.get(), "general.architecture", out.architecture, error)) {
        return false;
    }
    out.hidden_dim = 0;
    if (out.has_architecture) {
        const std::string hidden_dim_key = out.architecture + ".embedding_length";
        gguf_optional_u32(ctx.get(), hidden_dim_key.c_str(), out.hidden_dim);
    }
    out.tensor_count = gguf_get_n_tensors(ctx.get());
    out.first_layer = 0;
    out.last_layer_exclusive = std::numeric_limits<uint32_t>::max();
    if (gguf_optional_u32(ctx.get(), "split.count", out.legacy_split_count)) {
        out.has_legacy_split_metadata = true;
        gguf_optional_u32(ctx.get(), "split.no", out.legacy_split_no);
    }

    out.has_tensorrelay_stage_metadata = gguf_find_key(ctx.get(), "tensorrelay.stage") >= 0;
    if (!out.has_tensorrelay_stage_metadata) {
        return true;
    }

    if (!gguf_bool(ctx.get(), "tensorrelay.stage", out.stage_marker, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.index", out.stage_index, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.count", out.stage_count, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.first_layer", out.first_layer, error) ||
        !gguf_u32(ctx.get(), "tensorrelay.stage.last_layer_exclusive", out.last_layer_exclusive, error)) {
        return false;
    }
    return true;
}

static bool validate_stage_metadata(
        const tr_stage_metadata & meta,
        const tr_stage_executor_load_params * params,
        std::string & error) {
    if (meta.tensor_count <= 0) {
        error = "GGUF artifact contains no tensors";
        return false;
    }

    if (!meta.has_tensorrelay_stage_metadata) {
        if (meta.has_legacy_split_metadata) {
            error = "GGUF artifact is a legacy llama.cpp tensor-split shard without TensorRelay layer-stage metadata"
                " (split.no=" + std::to_string(meta.legacy_split_no) +
                " split.count=" + std::to_string(meta.legacy_split_count) +
                "); custom multi-node inference requires TensorRelay layer-stage GGUF artifacts";
            return false;
        }
        if (params->split_count == 1 && params->stage_index == 0) {
            if (!meta.has_architecture) {
                error = "missing GGUF metadata key: general.architecture";
                return false;
            }
            if (!is_supported_architecture(meta.architecture)) {
                error = "unsupported TensorRelay stage architecture: " + meta.architecture;
                return false;
            }
            return true;
        }
        error = "GGUF artifact is missing TensorRelay layer-stage metadata; custom multi-node inference requires"
                " TensorRelay layer-stage GGUF artifacts";
        return false;
    }

    if (!meta.has_architecture) {
        error = "missing GGUF metadata key: general.architecture";
        return false;
    }
    if (!is_supported_architecture(meta.architecture)) {
        error = "unsupported TensorRelay stage architecture: " + meta.architecture;
        return false;
    }
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
    if (meta.hidden_dim == 0) {
        error = "GGUF artifact is missing architecture embedding length metadata";
        return false;
    }
    return true;
}

static bool stage_owns_boundary(
        const tr_stage_executor_state * state,
        int32_t source_stage,
        int32_t target_stage) {
    if (state == nullptr || !state->loaded || state->split_count <= 1) {
        return false;
    }
    if (target_stage != source_stage + 1) {
        return false;
    }
    if (source_stage < 0 || target_stage >= state->split_count) {
        return false;
    }
    return state->stage_index == source_stage || state->stage_index == target_stage;
}

static bool build_activation_descriptor_json(
        tr_stage_executor_state * state,
        int32_t source_stage,
        int32_t target_stage,
        std::string & out) {
    if (state == nullptr) {
        return false;
    }
    if (!stage_owns_boundary(state, source_stage, target_stage)) {
        set_error(
            state,
            "stage executor does not own the requested adjacent activation boundary",
            TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        return false;
    }
    if (state->architecture.empty() || state->hidden_dim == 0) {
        set_error(
            state,
            "stage executor cannot describe boundary without loaded architecture metadata",
            TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED);
        return false;
    }

    json descriptor = {
        { "descriptor_version", 1 },
        { "source_stage", source_stage },
        { "target_stage", target_stage },
        { "architecture", state->architecture },
        { "dtype", "f16" },
        { "shape", json::array({ 0, state->hidden_dim }) },
        { "dynamic_dims", json::array({ 0 }) },
        { "layout", "row_major_token_hidden" },
        { "encoding", "raw_fp16" },
        { "max_chunk_size", 256 * 1024 },
    };
    out = descriptor.dump();
    return true;
}

static bool build_capabilities_json(tr_stage_executor_state * state, std::string & out) {
    if (state == nullptr) {
        return false;
    }
    if (!state->loaded) {
        set_error(
            state,
            "stage executor cannot report capabilities before load",
            TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED);
        return false;
    }

    const bool is_first_stage = state->stage_index == 0;
    const bool is_final_stage = state->stage_index == state->split_count - 1;
    const bool single_stage_request_json =
            state->split_count == 1 &&
            state->model != nullptr &&
            state->ctx != nullptr &&
            state->vocab != nullptr;
    const bool split_arch_supported = is_split_graph_execution_architecture(state->architecture);
    const bool split_graph_execution =
            state->split_count > 1 &&
            split_arch_supported &&
            state->model != nullptr &&
            state->ctx != nullptr;
    const std::string reason = [&]() {
        if (single_stage_request_json) {
            return std::string("single-stage request-json execution is available");
        }
        if (state->split_count <= 1) {
            return std::string("single-stage llama runtime is not initialized");
        }
        if (!split_arch_supported) {
            return "native split graph execution currently supports qwen2, qwen2moe, qwen3, and qwen3moe; architecture "
                + state->architecture + " is metadata-only in this build";
        }
        if (split_graph_execution) {
            return std::string("native split graph execution is available");
        }
        return std::string("native split graph execution runtime is not initialized");
    }();

    json capabilities = {
        { "capabilities_version", 1 },
        { "abi_version", TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION },
        { "stage_index", state->stage_index },
        { "split_count", state->split_count },
        { "first_layer", state->first_layer },
        { "last_layer_exclusive", state->last_layer_exclusive },
        { "architecture", state->architecture },
        { "hidden_dim", state->hidden_dim },
        { "slot_count", state->slot_count },
        { "context_length", state->context_length },
        { "devices", state->device_names },
        { "is_first_stage", is_first_stage },
        { "is_final_stage", is_final_stage },
        { "single_stage_request_json", single_stage_request_json },
        { "split_graph_execution", split_graph_execution },
        { "token_input", state->split_count > 1 && is_first_stage },
        { "activation_input", state->split_count > 1 && !is_first_stage },
        { "activation_output", state->split_count > 1 && !is_final_stage },
        { "sampled_token_output", state->split_count > 1 && is_final_stage },
        { "final_text_output", single_stage_request_json || (split_graph_execution && is_final_stage) },
        { "reason", reason },
    };
    out = capabilities.dump();
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

struct split_batch_storage {
    std::vector<llama_token> tokens;
    std::vector<float> embeddings;
    std::vector<llama_pos> positions;
    std::vector<int32_t> n_seq_ids;
    std::vector<llama_seq_id> seq_ids;
    std::vector<llama_seq_id *> seq_id_ptrs;
    std::vector<int8_t> logits;
    std::vector<uint32_t> output_ordinals;
    llama_batch batch{};

    bool reset(
            tr_stage_executor_state * state,
            const tr_stage_executor_batch_call * call,
            std::string & error) {
        const bool is_first_stage = state->stage_index == 0;
        const bool is_final_stage = state->stage_index == state->split_count - 1;
        size_t total_tokens = 0;
        for (uint32_t i = 0; i < call->input_count; ++i) {
            const auto & input = call->inputs_ptr[i];
            if (input.token_count == 0) {
                error = "split-stage input token_count must be greater than zero";
                return false;
            }
            total_tokens += input.token_count;
        }
        if (total_tokens > std::numeric_limits<int32_t>::max()) {
            error = "split-stage batch is too large";
            return false;
        }

        tokens.clear();
        embeddings.clear();
        positions.resize(total_tokens);
        n_seq_ids.assign(total_tokens, 1);
        seq_ids.resize(total_tokens);
        seq_id_ptrs.resize(total_tokens);
        logits.assign(total_tokens, 0);
        output_ordinals.clear();
        if (is_first_stage) {
            tokens.resize(total_tokens);
        } else {
            embeddings.resize(total_tokens * state->hidden_dim);
        }

        size_t cursor = 0;
        uint32_t output_ordinal = 0;
        for (uint32_t i = 0; i < call->input_count; ++i) {
            const auto & input = call->inputs_ptr[i];
            const llama_seq_id seq_id = static_cast<llama_seq_id>(input.slot_id);

            if (is_first_stage) {
                const uint64_t token_span_len =
                    static_cast<uint64_t>(input.token_count) * sizeof(llama_token);
                const uint64_t token_span_end = static_cast<uint64_t>(input.token_offset) + token_span_len;
                if ((input.flags & TENSORRELAY_STAGE_EXECUTOR_INPUT_FLAG_TOKEN_IDS) == 0) {
                    error = "first split stage requires token-id input";
                    return false;
                }
                if (token_span_end > call->input_bytes_len) {
                    error = "token-id input span is outside input buffer";
                    return false;
                }
                std::memcpy(
                    tokens.data() + cursor,
                    call->input_bytes_ptr + input.token_offset,
                    static_cast<size_t>(token_span_len));
            } else {
                const uint64_t expected_len =
                    static_cast<uint64_t>(input.token_count) * state->hidden_dim * sizeof(ggml_fp16_t);
                const uint64_t activation_span_end =
                    static_cast<uint64_t>(input.activation_offset) + static_cast<uint64_t>(input.activation_len);
                if ((input.flags & TENSORRELAY_STAGE_EXECUTOR_INPUT_FLAG_ACTIVATIONS) == 0) {
                    error = "non-first split stage requires activation input";
                    return false;
                }
                if (input.activation_len != expected_len) {
                    error = "activation input length does not match token_count * hidden_dim * sizeof(f16)";
                    return false;
                }
                if (activation_span_end > call->input_bytes_len) {
                    error = "activation input span is outside input buffer";
                    return false;
                }
                const auto * src = reinterpret_cast<const ggml_fp16_t *>(
                    call->input_bytes_ptr + input.activation_offset);
                float * dst = embeddings.data() + cursor * state->hidden_dim;
                ggml_fp16_to_fp32_row(src, dst, input.token_count * state->hidden_dim);
            }

            for (uint32_t j = 0; j < input.token_count; ++j) {
                const size_t idx = cursor + j;
                positions[idx] = static_cast<llama_pos>(input.position_start + j);
                seq_ids[idx] = seq_id;
                seq_id_ptrs[idx] = &seq_ids[idx];
            }

            if (is_final_stage) {
                logits[cursor + input.token_count - 1] = 1;
                output_ordinals.push_back(output_ordinal++);
            } else {
                for (uint32_t j = 0; j < input.token_count; ++j) {
                    logits[cursor + j] = 1;
                    output_ordinal++;
                }
            }
            cursor += input.token_count;
        }

        batch = {};
        batch.n_tokens = static_cast<int32_t>(total_tokens);
        batch.token = is_first_stage ? tokens.data() : nullptr;
        batch.embd = is_first_stage ? nullptr : embeddings.data();
        batch.pos = positions.data();
        batch.n_seq_id = n_seq_ids.data();
        batch.seq_id = seq_id_ptrs.data();
        batch.logits = logits.data();
        return true;
    }
};

static llama_sampler * create_default_sampler() {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return sampler;
}

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

    llama_sampler * sampler = create_default_sampler();
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

static bool execute_split_stage(
        tr_stage_executor_state * state,
        tr_stage_executor_batch_call * call,
        std::string & error) {
    if (state->model == nullptr || state->ctx == nullptr) {
        error = "split-stage llama runtime is not initialized";
        return false;
    }
    if (!is_split_graph_execution_architecture(state->architecture)) {
        error = "split-stage graph execution is not available for architecture " + state->architecture;
        return false;
    }

    const bool is_final_stage = state->stage_index == state->split_count - 1;
    size_t output_required = 0;
    for (uint32_t i = 0; i < call->input_count; ++i) {
        const auto & input = call->inputs_ptr[i];
        if (input.request_epoch < state->slot_epochs[input.slot_id]) {
            error = "split-stage input references a stale slot epoch";
            return false;
        }
        if (input.request_epoch > state->slot_epochs[input.slot_id]) {
            llama_memory_seq_rm(llama_get_memory(state->ctx), static_cast<llama_seq_id>(input.slot_id), -1, -1);
            state->slot_epochs[input.slot_id] = input.request_epoch;
        }
        if (!is_final_stage) {
            output_required += static_cast<size_t>(input.token_count) * state->hidden_dim * sizeof(ggml_fp16_t);
        }
    }
    if (!is_final_stage && output_required > call->output_bytes_len) {
        error = "batch output buffer is too small for activation output";
        return false;
    }

    split_batch_storage batch;
    if (!batch.reset(state, call, error)) {
        return false;
    }

    const int decode_rc = llama_decode(state->ctx, batch.batch);
    if (decode_rc != 0) {
        error = "llama_decode failed with code " + std::to_string(decode_rc);
        return false;
    }

    if (is_final_stage) {
        size_t output_cursor = 0;
        for (uint32_t i = 0; i < call->input_count; ++i) {
            auto & output = call->outputs_ptr[i];
            llama_sampler * sampler = create_default_sampler();
            std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_guard(sampler, llama_sampler_free);
            const llama_token next_token = llama_sampler_sample(
                sampler,
                state->ctx,
                static_cast<int32_t>(batch.output_ordinals[i]));

            const bool is_eog = state->vocab != nullptr && llama_vocab_is_eog(state->vocab, next_token);
            std::string piece;
            if (state->vocab != nullptr && !is_eog) {
                if (!token_to_piece(state->vocab, next_token, piece, error)) {
                    return false;
                }
            }
            if (output_cursor + piece.size() > call->output_bytes_len) {
                error = "batch output buffer is too small for sampled token text";
                return false;
            }
            if (!piece.empty()) {
                std::memcpy(call->output_bytes_ptr + output_cursor, piece.data(), piece.size());
            }

            output.slot_id = call->inputs_ptr[i].slot_id;
            output.request_epoch = call->inputs_ptr[i].request_epoch;
            output.activation_offset = static_cast<uint32_t>(output_cursor);
            output.activation_len = static_cast<uint32_t>(piece.size());
            output.sampled_token_id = next_token;
            output.flags = TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_TOKEN_ID |
                TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_FINAL_TEXT |
                (is_eog ? TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_EOG : 0u);
            output_cursor += piece.size();
        }
        return true;
    }

    float * embeddings = llama_get_embeddings(state->ctx);
    if (embeddings == nullptr) {
        error = "llama did not return split-stage activation embeddings";
        return false;
    }

    size_t output_cursor = 0;
    size_t embedding_cursor = 0;
    for (uint32_t i = 0; i < call->input_count; ++i) {
        const auto & input = call->inputs_ptr[i];
        auto & output = call->outputs_ptr[i];
        const size_t float_count = static_cast<size_t>(input.token_count) * state->hidden_dim;
        const size_t byte_count = float_count * sizeof(ggml_fp16_t);
        auto * dst = reinterpret_cast<ggml_fp16_t *>(call->output_bytes_ptr + output_cursor);
        ggml_fp32_to_fp16_row(embeddings + embedding_cursor, dst, float_count);

        output.slot_id = input.slot_id;
        output.request_epoch = input.request_epoch;
        output.activation_offset = static_cast<uint32_t>(output_cursor);
        output.activation_len = static_cast<uint32_t>(byte_count);
        output.sampled_token_id = -1;
        output.flags = TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_ACTIVATIONS;

        output_cursor += byte_count;
        embedding_cursor += float_count;
    }
    return true;
}

static std::vector<std::string> split_selected_device_names(const std::string & selected_devices) {
    static const char * whitespace = " \t\r\n";
    std::vector<std::string> names;
    size_t start = 0;
    while (start <= selected_devices.size()) {
        size_t end = selected_devices.find(',', start);
        if (end == std::string::npos) {
            end = selected_devices.size();
        }
        const std::string segment = selected_devices.substr(start, end - start);
        const size_t first = segment.find_first_not_of(whitespace);
        if (first != std::string::npos) {
            const size_t last = segment.find_last_not_of(whitespace);
            names.push_back(segment.substr(first, last - first + 1));
        }
        start = end + 1;
    }
    return names;
}

static std::string available_gpu_device_names() {
    std::string names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (!names.empty()) {
            names += ", ";
        }
        names += ggml_backend_dev_name(dev);
    }
    return names.empty() ? std::string("none") : names;
}

// Resolves the comma-separated device-name allowlist into backend device handles.
// Must be called after ggml_backend_load_all(). On success out_devices is either
// empty (CPU-only) or a NULL-terminated array of GPU device handles.
static bool resolve_selected_devices(
        const std::string & selected_devices,
        std::vector<ggml_backend_dev_t> & out_devices,
        std::vector<std::string> & out_names,
        std::string & error) {
    out_devices.clear();
    out_names.clear();
    const std::vector<std::string> requested = split_selected_device_names(selected_devices);
    if (requested.empty()) {
        return true;
    }
    for (const auto & name : requested) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (dev == nullptr) {
            error = "selected device not found: " + name +
                " (available devices: " + available_gpu_device_names() + ")";
            return false;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            error = "selected device is not a GPU device: " + name +
                " (available devices: " + available_gpu_device_names() + ")";
            return false;
        }
        out_devices.push_back(dev);
        out_names.push_back(ggml_backend_dev_name(dev));
    }
    out_devices.push_back(nullptr);
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
    state->hidden_dim = metadata.hidden_dim;
    state->slot_count = params->slot_count;
    state->context_length = params->context_length;
    state->vocab = nullptr;
    state->device_names.clear();

    const bool load_llama_runtime =
        params->split_count == 1 || is_split_graph_execution_architecture(metadata.architecture);
    if (load_llama_runtime) {
        std::call_once(backend_init_once, []() {
            llama_backend_init();
            ggml_backend_load_all();
        });
        std::vector<ggml_backend_dev_t> devices;
        std::string device_error;
        if (!resolve_selected_devices(state->selected_devices, devices, state->device_names, device_error)) {
            return set_error(state, device_error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        llama_model_params model_params = llama_model_default_params();
        if (devices.empty()) {
            model_params.n_gpu_layers = 0;
        } else {
            // with the default LLAMA_SPLIT_MODE_LAYER, llama_model_load_from_file copies
            // the device handles out of this NULL-terminated array, so the local vector
            // only has to stay alive for the duration of the call
            model_params.n_gpu_layers = 999;
            model_params.devices = devices.data();
        }
        state->model = llama_model_load_from_file(state->shard_path.c_str(), model_params);
        if (state->model == nullptr) {
            return set_error(state, "failed to load llama model", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
        state->vocab = llama_model_get_vocab(state->model);
        if (state->vocab == nullptr) {
            return set_error(state, "llama model has no vocabulary", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = state->context_length > 0 ? state->context_length : 4096;
        ctx_params.n_batch = std::min<uint32_t>(ctx_params.n_ctx, 512);
        ctx_params.n_seq_max = std::max<uint32_t>(state->slot_count, 1);
        ctx_params.no_perf = true;
        ctx_params.embeddings = params->split_count > 1 && params->stage_index < params->split_count - 1;
        ctx_params.n_outputs_max = ctx_params.n_batch;
        state->ctx = llama_init_from_model(state->model, ctx_params);
        if (state->ctx == nullptr) {
            llama_model_free(state->model);
            state->model = nullptr;
            state->vocab = nullptr;
            return set_error(state, "failed to create llama context", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
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

size_t tr_stage_executor_activation_descriptor_json(
        void * handle,
        int32_t source_stage,
        int32_t target_stage,
        uint8_t * out_ptr,
        size_t out_len) {
    auto * state = as_state(handle);
    if (state == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    std::string descriptor;
    if (!build_activation_descriptor_json(state, source_stage, target_stage, descriptor)) {
        return 0;
    }
    if (out_ptr == nullptr || out_len == 0) {
        return descriptor.size();
    }
    const size_t n = std::min(out_len, descriptor.size());
    std::memcpy(out_ptr, descriptor.data(), n);
    return n;
}

size_t tr_stage_executor_capabilities_json(
        void * handle,
        uint8_t * out_ptr,
        size_t out_len) {
    auto * state = as_state(handle);
    if (state == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    std::string capabilities;
    if (!build_capabilities_json(state, capabilities)) {
        return 0;
    }
    if (out_ptr == nullptr || out_len == 0) {
        return capabilities.size();
    }
    const size_t n = std::min(out_len, capabilities.size());
    std::memcpy(out_ptr, capabilities.data(), n);
    return n;
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
    if (call->input_count > 0 && call->input_bytes_ptr == nullptr) {
        return set_error(state, "batch input byte buffer is required", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
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
        std::string error;
        if (!execute_split_stage(state, call, error)) {
            const int32_t code = is_split_graph_execution_architecture(state->architecture)
                ? TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT
                : TENSORRELAY_STAGE_EXECUTOR_ERR_UNSUPPORTED;
            return set_error(state, error, code);
        }
        state->last_error.clear();
        return TENSORRELAY_STAGE_EXECUTOR_OK;
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

// ABI v4 tokenization entry point. Any stage GGUF carries the full tokenizer/chat-template
// KV map, so every runtime node can tokenize a normalized OpenAI request via its local
// executor even when its stage does not own token embeddings or output tensors.
int32_t tr_stage_executor_tokenize(
        void * handle,
        const uint8_t * request_json_ptr,
        size_t request_json_len,
        int32_t * out_tokens_ptr,
        size_t out_tokens_cap,
        size_t * out_token_count) {
    auto * state = as_state(handle);
    if (state == nullptr || out_token_count == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    *out_token_count = 0;
    if (request_json_ptr == nullptr || request_json_len == 0) {
        return set_error(state, "tokenize request payload is empty", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->loaded || state->model == nullptr || state->vocab == nullptr) {
        return set_error(
            state,
            "tokenize requires a loaded llama model with vocabulary",
            TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED);
    }

    const std::string payload = bytes_to_string(request_json_ptr, request_json_len);
    std::string prompt;
    std::string error;
    int max_tokens = 0;
    if (!request_to_prompt(state->model, payload, prompt, max_tokens, error)) {
        return set_error(state, error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }

    const int n_prompt = -llama_tokenize(
        state->vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        return set_error(state, "failed to tokenize prompt", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    *out_token_count = static_cast<size_t>(n_prompt);
    if (out_tokens_ptr == nullptr || out_tokens_cap < static_cast<size_t>(n_prompt)) {
        // probe call: report required capacity without writing token ids
        state->last_error.clear();
        return TENSORRELAY_STAGE_EXECUTOR_OK;
    }

    static_assert(sizeof(llama_token) == sizeof(int32_t), "llama_token must remain 32-bit for the tokenize ABI");
    if (llama_tokenize(
            state->vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            reinterpret_cast<llama_token *>(out_tokens_ptr),
            static_cast<int32_t>(out_tokens_cap),
            true,
            true) < 0) {
        *out_token_count = 0;
        return set_error(state, "failed to tokenize prompt", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
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
