#include "tensorrelay_stage_executor.h"

#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

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
};

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
    delete as_state(handle);
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

    state->shard_path = shard_path;
    state->selected_devices = bytes_to_string(params->selected_devices_ptr, params->selected_devices_len);
    state->architecture = metadata.architecture;
    state->stage_index = params->stage_index;
    state->split_count = params->split_count;
    state->first_layer = metadata.first_layer;
    state->last_layer_exclusive = metadata.last_layer_exclusive;
    state->slot_count = params->slot_count;
    state->context_length = params->context_length;
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

    size_t output_cursor = 0;
    for (uint32_t i = 0; i < call->input_count; ++i) {
        const auto & input = call->inputs_ptr[i];
        auto & output = call->outputs_ptr[i];
        const auto * input_begin = call->input_bytes_ptr + input.activation_offset;
        const std::string input_payload(reinterpret_cast<const char *>(input_begin), input.activation_len);
        std::string payload;

        if (state->stage_index >= state->split_count - 1) {
            payload =
                "TensorRelay diagnostic response: activation stream reached final stage " +
                std::to_string(state->stage_index) +
                ". Native llama graph-boundary execution is still pending in this fork.";
            output.flags = TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_FINAL_TEXT;
            output.sampled_token_id = -1;
        } else {
            payload =
                "tensorrelay-diagnostic-activation\nstage=" +
                std::to_string(state->stage_index) +
                "\nnext_stage=" +
                std::to_string(state->stage_index + 1) +
                "\narchitecture=" +
                state->architecture +
                "\ninput_bytes=" +
                std::to_string(input_payload.size()) +
                "\n";
            output.flags = 0;
            output.sampled_token_id = -1;
        }

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
    if (slot_id >= state->slot_epochs.size()) {
        return set_error(state, "release references an unknown slot", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
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
