#include "tensorrelay_stage_executor.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

    // Architecture layout metadata used by the per-model split soundness checks.
    uint32_t    block_count = 0;        // <arch>.block_count (includes NextN/MTP tail blocks)
    uint32_t    nextn_layers = 0;       // <arch>.nextn_predict_layers (0 when absent)
    // gemma4
    uint32_t    per_layer_embd = 0;     // <arch>.embedding_length_per_layer_input
    uint32_t    shared_kv_layers = 0;   // <arch>.attention.shared_kv_layers
    std::vector<bool> swa_layers;       // gemma4: per-layer sliding-window flags
    // qwen35 / qwen35moe
    std::vector<bool> recurrent_layers; // per-layer linear-attention (gated delta net) flags
};

// Per-slot sampler chain installed via tr_stage_executor_configure_slot_sampler.
// The chain persists across sampling steps of one (slot, request epoch) so
// stateful samplers (dist RNG) advance correctly over a generation.
struct tr_slot_sampler_state {
    uint64_t        request_epoch = 0;
    bool            configured = false;
    llama_sampler * sampler = nullptr;
};

struct tr_stage_executor_state {
    std::string last_error;
    std::string shard_path;
    std::string selected_devices;
    std::string architecture;
    // Non-empty when the architecture is split-capable in general but this
    // specific model/stage layout cannot be executed with hidden-state-only
    // boundaries (e.g. gemma4 per-layer embeddings or split KV-shared layers).
    std::string split_unsound_reason;
    int32_t     stage_index = -1;
    int32_t     split_count = 0;
    uint32_t    first_layer = 0;
    uint32_t    last_layer_exclusive = 0;
    uint32_t    hidden_dim = 0;
    uint32_t    block_count = 0;
    uint32_t    slot_count = 0;
    uint32_t    context_length = 0;
    uint32_t    pos_streams = 1;
    // ABI v8 tail-spill; retained for capabilities reporting.
    uint32_t    spill_layer_count = 0;
    std::string spill_devices;
    std::string spill_engine_applied;
    bool        loaded = false;
    std::vector<std::string> device_names;
    std::vector<uint64_t> slot_epochs;
    std::vector<tr_slot_sampler_state> slot_samplers;
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

// Reads a per-layer boolean layer-type pattern that converters store either as
// a scalar (broadcast to every layer) or as an array of bool/int values.
// Returns false when the key is absent or has an unusable type.
static bool gguf_optional_layer_flags(
        const gguf_context * ctx,
        const char * key,
        uint32_t n_layer,
        std::vector<bool> & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || n_layer == 0) {
        return false;
    }
    const gguf_type type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_BOOL) {
        out.assign(n_layer, gguf_get_val_bool(ctx, id));
        return true;
    }
    if (type != GGUF_TYPE_ARRAY) {
        uint32_t value = 0;
        if (!read_gguf_u32_value(ctx, id, value)) {
            return false;
        }
        out.assign(n_layer, value != 0);
        return true;
    }
    const size_t n = gguf_get_arr_n(ctx, id);
    if (n == 0) {
        return false;
    }
    const gguf_type elem = gguf_get_arr_type(ctx, id);
    const void * data = gguf_get_arr_data(ctx, id);
    if (data == nullptr) {
        return false;
    }
    out.assign(n_layer, false);
    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t src = std::min<size_t>(il, n - 1);
        switch (elem) {
            case GGUF_TYPE_BOOL:
                out[il] = static_cast<const int8_t *>(data)[src] != 0;
                break;
            case GGUF_TYPE_INT8:
                out[il] = static_cast<const int8_t *>(data)[src] != 0;
                break;
            case GGUF_TYPE_UINT8:
                out[il] = static_cast<const uint8_t *>(data)[src] != 0;
                break;
            case GGUF_TYPE_INT16:
                out[il] = static_cast<const int16_t *>(data)[src] != 0;
                break;
            case GGUF_TYPE_UINT16:
                out[il] = static_cast<const uint16_t *>(data)[src] != 0;
                break;
            case GGUF_TYPE_INT32:
                out[il] = static_cast<const int32_t *>(data)[src] != 0;
                break;
            case GGUF_TYPE_UINT32:
                out[il] = static_cast<const uint32_t *>(data)[src] != 0;
                break;
            default:
                return false;
        }
    }
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

// Architectures whose graph builders support TensorRelay layer-stage execution
// (hidden-state-only stage boundaries). qwen35/qwen35moe carry per-layer
// recurrent (gated delta net) state and gemma4 carries per-layer iSWA KV state,
// but both keep that state strictly layer-local, so it stays inside the stage
// that owns the layer. Some gemma4 configurations are still unsound to split
// (per-layer token embeddings, split KV-shared tails); those are rejected
// per-model by split_stage_unsound_reason() below.
static bool is_split_graph_execution_architecture(const std::string & architecture) {
    static const std::set<std::string> supported = {
        "qwen2",
        "qwen2moe",
        "qwen3",
        "qwen3moe",
        "qwen35",
        "qwen35moe",
        "gemma4",
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
        const std::string arch = out.architecture;
        gguf_optional_u32(ctx.get(), (arch + ".embedding_length").c_str(), out.hidden_dim);
        gguf_optional_u32(ctx.get(), (arch + ".block_count").c_str(), out.block_count);
        gguf_optional_u32(ctx.get(), (arch + ".nextn_predict_layers").c_str(), out.nextn_layers);
        // main-pass layer count (NextN/MTP tail blocks are not executed by the executor)
        const uint32_t n_layer = out.block_count > out.nextn_layers ? out.block_count - out.nextn_layers : 0;
        if (arch == "gemma4") {
            gguf_optional_u32(ctx.get(), (arch + ".embedding_length_per_layer_input").c_str(), out.per_layer_embd);
            gguf_optional_u32(ctx.get(), (arch + ".attention.shared_kv_layers").c_str(), out.shared_kv_layers);
            gguf_optional_layer_flags(
                ctx.get(), (arch + ".attention.sliding_window_pattern").c_str(), n_layer, out.swa_layers);
        }
        if (arch == "qwen35" || arch == "qwen35moe") {
            if (!gguf_optional_layer_flags(
                    ctx.get(), (arch + ".attention.recurrent_layers").c_str(), n_layer, out.recurrent_layers)) {
                // default layout used by llama.cpp when the explicit array is absent:
                // every layer is linear attention except each full_attention_interval-th
                uint32_t full_attn_interval = 4;
                gguf_optional_u32(ctx.get(), (arch + ".full_attention_interval").c_str(), full_attn_interval);
                if (full_attn_interval > 0 && n_layer > 0) {
                    out.recurrent_layers.assign(n_layer, false);
                    for (uint32_t il = 0; il < n_layer; ++il) {
                        out.recurrent_layers[il] = (il + 1) % full_attn_interval != 0;
                    }
                }
            }
        }
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

// Per-model split soundness check for architectures whose graphs are only
// splittable at hidden-state-only boundaries under extra layout constraints.
// Returns a non-empty human-readable reason when THIS stage cannot soundly run
// native split graph execution; empty string means the stage layout is sound.
static std::string split_stage_unsound_reason(const tr_stage_metadata & meta) {
    const uint32_t n_layer = meta.block_count > meta.nextn_layers ? meta.block_count - meta.nextn_layers : 0;
    const uint32_t first = meta.first_layer;
    const uint32_t last = std::min(meta.last_layer_exclusive, n_layer);

    if (n_layer == 0 || first >= last) {
        return "stage layer range contains no executable (non-NextN) layers";
    }

    if (meta.architecture == "gemma4") {
        if (meta.per_layer_embd != 0) {
            return "gemma4 per-layer token embeddings require the token ids and the layer-0 input"
                   " embedding at every layer, which do not cross hidden-state-only stage boundaries";
        }
        if (meta.shared_kv_layers > 0 && meta.shared_kv_layers < n_layer) {
            // Shared tail layers reuse the K/V written by the last two KV-owning
            // layers (n_layer_kv_from_start - 1 and - 2); a stage that executes a
            // shared layer must also execute both of its KV source layers.
            const uint32_t kv_from_start = n_layer - meta.shared_kv_layers;
            if (last > kv_from_start && (kv_from_start < 2 || first > kv_from_start - 2)) {
                return "gemma4 stage layer range separates KV-shared layers from the source layers"
                       " whose K/V they reuse";
            }
        }
        // The iSWA graph always builds both the sliding-window and the full
        // attention mask inputs; a stage lacking one layer kind would leave the
        // other mask without a consumer (and without an allocated buffer).
        if (!meta.swa_layers.empty()) {
            bool has_swa = false;
            bool has_full = false;
            for (uint32_t il = first; il < last; ++il) {
                (meta.swa_layers[il] ? has_swa : has_full) = true;
            }
            bool model_has_swa = false;
            bool model_has_full = false;
            for (uint32_t il = 0; il < n_layer; ++il) {
                (meta.swa_layers[il] ? model_has_swa : model_has_full) = true;
            }
            if ((model_has_swa && !has_swa) || (model_has_full && !has_full)) {
                return "gemma4 stage layer range must contain at least one sliding-window and one"
                       " full-attention layer so that both attention mask inputs stay consumed";
            }
        }
    }

    if (meta.architecture == "qwen35" || meta.architecture == "qwen35moe") {
        // The hybrid graph always builds both the recurrent-state and the
        // attention KV inputs; a stage lacking one layer kind would leave the
        // other input without a consumer (and without an allocated buffer).
        if (meta.recurrent_layers.empty()) {
            return "qwen35 stage metadata is missing the recurrent layer layout";
        }
        bool has_recr = false;
        bool has_attn = false;
        for (uint32_t il = first; il < last; ++il) {
            (meta.recurrent_layers[il] ? has_recr : has_attn) = true;
        }
        if (!has_recr || !has_attn) {
            return "qwen35 stage layer range must contain at least one linear-attention and one"
                   " full-attention layer so that both hybrid memory inputs stay consumed";
        }
    }

    return {};
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
    const bool split_arch_supported =
            is_split_graph_execution_architecture(state->architecture) &&
            state->split_unsound_reason.empty();
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
        if (!is_split_graph_execution_architecture(state->architecture)) {
            return "native split graph execution currently supports qwen2, qwen2moe, qwen3, qwen3moe,"
                " qwen35, qwen35moe, and gemma4; architecture "
                + state->architecture + " is metadata-only in this build";
        }
        if (!state->split_unsound_reason.empty()) {
            return state->split_unsound_reason;
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
        // ABI v8: build capability plus the current load's applied spill echo.
        { "supports_layer_spill", true },
        { "spill_layer_count", state->spill_layer_count },
        { "spill_devices", state->spill_engine_applied },
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
        // M-RoPE models read one position stream per RoPE section from embedding
        // batches (token batches broadcast stream 0 internally); fill every
        // stream with the linear text position.
        const uint32_t n_pos_streams = std::max<uint32_t>(state->pos_streams, 1);
        positions.resize(total_tokens * n_pos_streams);
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
                const llama_pos pos = static_cast<llama_pos>(input.position_start + j);
                for (uint32_t s = 0; s < n_pos_streams; ++s) {
                    positions[s * total_tokens + idx] = pos;
                }
                seq_ids[idx] = seq_id;
                seq_id_ptrs[idx] = &seq_ids[idx];
            }

            if (is_final_stage) {
                logits[cursor + input.token_count - 1] = 1;
                // llama_get_logits_ith() takes a batch token index (translated
                // through output_ids), not an output ordinal: record the index
                // of each input's sampled (last) token.
                output_ordinals.push_back(static_cast<uint32_t>(cursor + input.token_count - 1));
                output_ordinal++;
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

// Builds the sampler chain for one request's normalized sampling params.
// temperature <= 0 is the greedy contract: a single deterministic argmax
// sampler with no truncation or RNG in the chain.
static llama_sampler * create_sampler_from_params(const tr_stage_executor_sampler_params & params) {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        return sampler;
    }
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.min_p > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(params.min_p, 1));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));
    return sampler;
}

static void reset_slot_sampler(tr_slot_sampler_state & slot) {
    if (slot.sampler != nullptr) {
        llama_sampler_free(slot.sampler);
        slot.sampler = nullptr;
    }
    slot.configured = false;
    slot.request_epoch = 0;
}

static void reset_all_slot_samplers(tr_stage_executor_state * state) {
    for (auto & slot : state->slot_samplers) {
        reset_slot_sampler(slot);
    }
}

// Returns the sampler configured for this (slot, request epoch), or nullptr
// when the slot has no matching configuration (callers fall back to the
// default chain, preserving pre-ABI-v5 behavior).
static llama_sampler * configured_slot_sampler(
        tr_stage_executor_state * state,
        uint32_t slot_id,
        uint64_t request_epoch) {
    if (slot_id >= state->slot_samplers.size()) {
        return nullptr;
    }
    auto & slot = state->slot_samplers[slot_id];
    if (!slot.configured || slot.request_epoch != request_epoch) {
        return nullptr;
    }
    return slot.sampler;
}

// Extracts the per-request sampling parameters from an OpenAI request body.
// Absent fields keep the executor's historical defaults (min_p 0.05, temp 0.8,
// random seed), so requests without sampling fields behave exactly as before.
// temperature <= 0 selects deterministic greedy decoding, matching the
// split-stage tr_stage_executor_configure_slot_sampler contract.
static tr_stage_executor_sampler_params single_stage_sampler_params(const json & body) {
    tr_stage_executor_sampler_params params = {};
    params.abi_version = TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION;
    params.temperature = 0.8f;
    params.top_p       = 1.0f;
    params.min_p       = 0.05f;
    params.top_k       = 0;
    params.seed        = LLAMA_DEFAULT_SEED;

    const auto read_float = [&](const char * key, float current) {
        if (!body.contains(key) || !body.at(key).is_number()) {
            return current;
        }
        const float value = body.at(key).get<float>();
        return std::isfinite(value) ? value : current;
    };
    params.temperature = read_float("temperature", params.temperature);
    params.top_p       = read_float("top_p", params.top_p);
    params.min_p       = read_float("min_p", params.min_p);
    if (body.contains("top_k") && body.at("top_k").is_number_integer()) {
        params.top_k = body.at("top_k").get<int32_t>();
    }
    if (body.contains("seed") && body.at("seed").is_number_integer()) {
        const int64_t seed = body.at("seed").get<int64_t>();
        if (seed >= 0) {
            params.seed = static_cast<uint32_t>(seed & 0xFFFFFFFFll);
        }
    }
    return params;
}

// OpenAI "stop": a single string or an array of strings; empty entries are
// ignored. Generated text is truncated at the earliest stop match.
static std::vector<std::string> request_stop_strings(const json & body) {
    std::vector<std::string> stops;
    if (!body.contains("stop")) {
        return stops;
    }
    const auto & stop = body.at("stop");
    if (stop.is_string()) {
        std::string value = stop.get<std::string>();
        if (!value.empty()) {
            stops.push_back(std::move(value));
        }
        return stops;
    }
    if (stop.is_array()) {
        for (const auto & item : stop) {
            if (!item.is_string()) {
                continue;
            }
            std::string value = item.get<std::string>();
            if (!value.empty()) {
                stops.push_back(std::move(value));
            }
        }
    }
    return stops;
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

    // request_to_prompt() already rejected malformed JSON, so this re-parse
    // only extracts the optional sampling/stop fields.
    const json body = json::parse(payload);
    const tr_stage_executor_sampler_params sampler_params = single_stage_sampler_params(body);
    const std::vector<std::string> stop_strings = request_stop_strings(body);

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

    llama_sampler * sampler = create_sampler_from_params(sampler_params);
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

        // Stop strings: truncate at the earliest match (a match may span token
        // boundaries, so search the accumulated text) and stop generating.
        size_t stop_pos = std::string::npos;
        for (const auto & stop : stop_strings) {
            const size_t search_from = out.size() >= piece.size() + stop.size() - 1
                ? out.size() - piece.size() - (stop.size() - 1)
                : 0;
            const size_t pos = out.find(stop, search_from);
            if (pos != std::string::npos && pos < stop_pos) {
                stop_pos = pos;
            }
        }
        if (stop_pos != std::string::npos) {
            out.erase(stop_pos);
            break;
        }

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
    if (!state->split_unsound_reason.empty()) {
        error = "split-stage graph execution is not available for this model/stage layout: " +
            state->split_unsound_reason;
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
            // Per-request sampling: use the chain installed for this
            // (slot, request epoch) at reservation time; fall back to the
            // default chain when none was configured.
            llama_sampler * sampler = configured_slot_sampler(
                state, call->inputs_ptr[i].slot_id, call->inputs_ptr[i].request_epoch);
            std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_guard(nullptr, llama_sampler_free);
            if (sampler == nullptr) {
                sampler_guard.reset(create_default_sampler());
                sampler = sampler_guard.get();
            }
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

// ABI v8: resolve the buffer type that spilled tail tensors are placed on. An
// empty selector means the CPU (system-RAM) buffer type; otherwise it names a
// single ggml device (e.g. an iGPU) whose default buffer type is used. Must be
// called after ggml_backend_load_all().
// ABI v8: resolves the spill selector. Empty or naming the CPU device selects
// CPU spill (out_dev nullptr, CPU buffer type); otherwise the named GPU device
// and its buffer type. Unknown names are errors. Must be called after
// ggml_backend_load_all().
static bool resolve_spill_target(
        const std::string & spill_devices,
        ggml_backend_dev_t & out_dev,
        ggml_backend_buffer_type_t & out_buft,
        std::string & error) {
    out_dev = nullptr;
    out_buft = ggml_backend_cpu_buffer_type();
    std::string trimmed = spill_devices;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
    if (trimmed.empty()) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(trimmed.c_str());
    if (dev == nullptr) {
        error = "spill device is not available: " + trimmed +
            " (available devices: " + available_gpu_device_names() + ")";
        return false;
    }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (buft == nullptr) {
        error = "spill device has no buffer type: " + trimmed;
        return false;
    }
    out_dev = dev;
    out_buft = buft;
    return true;
}

// ABI v8: override list pinning the stage's spilled tail onto spill_buft. A
// stage shard keeps the model's ORIGINAL block indices and declares the full
// model's block_count, so the tail is the last `spill_layer_count` blocks of
// [first_layer, last_layer_exclusive). On the final stage the output head
// follows the tail (ADR-0018); tied-embedding models have no output.* tensors
// (their lm_head is a duplicated token_embd, which llama keeps on the CPU
// input device - already system memory), so the head patterns match nothing
// there. When head_buft is non-null (GPU spill target in the compute set) the
// head blocks are pinned so the layer split cannot migrate them onto the
// spill device; token_embd needs no pin (llama always places it on the CPU
// input device). out_patterns owns the strings referenced by out_overrides;
// the pointers are taken only after every push, and both vectors must outlive
// llama_model_load_from_file. The list is NULL-terminated.
static void build_spill_overrides(
        uint32_t first_layer,
        uint32_t last_layer_exclusive,
        uint32_t spill_layer_count,
        bool is_final_stage,
        ggml_backend_buffer_type_t spill_buft,
        ggml_backend_buffer_type_t head_buft,
        std::vector<std::string> & out_patterns,
        std::vector<llama_model_tensor_buft_override> & out_overrides) {
    out_patterns.clear();
    out_overrides.clear();
    const uint32_t tail_start = last_layer_exclusive - spill_layer_count;
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (uint32_t blk = first_layer; blk < last_layer_exclusive; ++blk) {
        const bool spilled = blk >= tail_start;
        if (!spilled && head_buft == nullptr) {
            continue; // head stays on the discrete device via n_gpu_layers
        }
        out_patterns.push_back("^blk\\." + std::to_string(blk) + "\\.");
        bufts.push_back(spilled ? spill_buft : head_buft);
    }
    if (is_final_stage && spill_layer_count > 0) {
        out_patterns.push_back("^output\\.");
        bufts.push_back(spill_buft);
        out_patterns.push_back("^output_norm\\.");
        bufts.push_back(spill_buft);
    }
    out_overrides.reserve(out_patterns.size() + 1);
    for (size_t i = 0; i < out_patterns.size(); ++i) {
        out_overrides.push_back({ out_patterns[i].c_str(), bufts[i] });
    }
    out_overrides.push_back({ nullptr, nullptr });
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
        reset_all_slot_samplers(state);
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

    reset_all_slot_samplers(state);
    state->slot_samplers.clear();
    // A failed load below must not leave the handle looking loaded against a
    // freed model/ctx, or report the previous load's spill in capabilities.
    state->loaded = false;
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
    state->block_count = metadata.block_count;
    state->slot_count = params->slot_count;
    state->context_length = params->context_length;
    state->pos_streams = 1;
    state->vocab = nullptr;
    state->device_names.clear();
    state->spill_layer_count = 0;
    state->spill_devices = bytes_to_string(params->spill_devices_ptr, params->spill_devices_len);
    state->spill_engine_applied.clear();

    state->split_unsound_reason.clear();
    if (params->split_count > 1 && is_split_graph_execution_architecture(metadata.architecture)) {
        state->split_unsound_reason = split_stage_unsound_reason(metadata);
    }

    const bool load_llama_runtime =
        params->split_count == 1 ||
        (is_split_graph_execution_architecture(metadata.architecture) && state->split_unsound_reason.empty());
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
        // These must outlive llama_model_load_from_file: it reads the override
        // patterns (regex strings) and buft list during the load.
        std::vector<std::string> spill_patterns;
        std::vector<llama_model_tensor_buft_override> spill_overrides;
        if (devices.empty()) {
            if (params->spill_layer_count > 0) {
                return set_error(state,
                    "spill requested but no discrete devices are selected",
                    TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
            }
            model_params.n_gpu_layers = 0;
        } else {
            // with the default LLAMA_SPLIT_MODE_LAYER, llama_model_load_from_file copies
            // the device handles out of this NULL-terminated array, so the local vector
            // only has to stay alive for the duration of the call
            model_params.n_gpu_layers = 999;
            // ABI v8 tail-spill. A solver-assigned spill must be applied
            // exactly or the load must fail: a spill silently dropped or
            // clamped loads a placement the solver never costed.
            if (params->spill_layer_count > 0) {
                const bool have_stage_range =
                    state->last_layer_exclusive > state->first_layer &&
                    state->last_layer_exclusive <= state->block_count;
                if (!have_stage_range) {
                    return set_error(state,
                        "spill requested but the stage layer range metadata is missing or invalid",
                        TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
                }
                if (params->spill_layer_count > state->last_layer_exclusive - state->first_layer) {
                    return set_error(state,
                        "spill_layer_count exceeds the stage's layer count",
                        TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
                }
                std::string spill_error;
                ggml_backend_dev_t spill_dev = nullptr;
                ggml_backend_buffer_type_t spill_buft = nullptr;
                if (!resolve_spill_target(state->spill_devices, spill_dev, spill_buft, spill_error)) {
                    return set_error(state, spill_error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
                }
                ggml_backend_buffer_type_t head_buft = nullptr;
                if (spill_dev != nullptr) {
                    // A GPU spill target only holds weights if it also joins the
                    // compute device set (else ggml aborts on a tensor in a buffer
                    // no scheduled backend can run); the CPU backend is always
                    // present so a CPU tail needs no entry. Joining the set means
                    // the head must be pinned so the layer split cannot migrate it
                    // onto the spill device - which only supports a single head
                    // device, and the spill device must not already be selected
                    // (a duplicate handle double-counts in the layer split).
                    if (devices.size() != 2) {
                        return set_error(state,
                            "GPU spill requires exactly one selected head device",
                            TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
                    }
                    if (devices.front() == spill_dev) {
                        return set_error(state,
                            "spill device must not be one of the selected devices",
                            TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
                    }
                    head_buft = ggml_backend_dev_buffer_type(devices.front());
                    devices.back() = spill_dev; // overwrite the NULL terminator
                    devices.push_back(nullptr);
                }
                const bool is_final_stage = params->stage_index == params->split_count - 1;
                build_spill_overrides(
                    state->first_layer, state->last_layer_exclusive, params->spill_layer_count,
                    is_final_stage, spill_buft, head_buft, spill_patterns, spill_overrides);
                model_params.tensor_buft_overrides = spill_overrides.data();
                state->spill_layer_count = params->spill_layer_count;
                state->spill_engine_applied =
                    spill_dev != nullptr ? ggml_backend_dev_name(spill_dev) : "cpu";
            }
            // Set last: a GPU spill target may have push_back'd onto `devices`,
            // reallocating it, so bind the pointer only after the final layout.
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

        // M-RoPE models (e.g. qwen35/qwen35moe IMRoPE) consume one position per
        // RoPE section when a batch carries embeddings instead of token ids, so
        // split-stage activation batches must supply all position streams.
        const llama_rope_type rope_type = llama_model_rope_type(state->model);
        state->pos_streams =
            (rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
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
    reset_all_slot_samplers(state);
    state->slot_samplers.assign(slot_count, tr_slot_sampler_state{});
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
            const int32_t code =
                is_split_graph_execution_architecture(state->architecture) &&
                state->split_unsound_reason.empty()
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
    if (slot_id < state->slot_samplers.size()) {
        reset_slot_sampler(state->slot_samplers[slot_id]);
    }
    state->slot_epochs[slot_id] = std::max(state->slot_epochs[slot_id], request_epoch);
    state->last_error.clear();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

// One benchmark decode step: a single decode-shaped token (token-id input on
// the first/single stage, zero-filled fp32 embedding on later stages) at the
// given position on scratch sequence 0.
static bool benchmark_decode_step(
        tr_stage_executor_state * state,
        std::vector<float> & scratch_embd,
        std::vector<llama_pos> & scratch_pos,
        llama_pos pos,
        std::string & error) {
    const bool token_input = state->split_count == 1 || state->stage_index == 0;
    llama_token token = 0;
    if (token_input && state->vocab != nullptr) {
        const llama_token bos = llama_vocab_bos(state->vocab);
        if (bos >= 0) {
            token = bos;
        }
    }
    const uint32_t n_pos_streams = std::max<uint32_t>(state->pos_streams, 1);
    scratch_pos.assign(n_pos_streams, pos);
    int32_t n_seq_one = 1;
    llama_seq_id seq_zero = 0;
    llama_seq_id * seq_ptr = &seq_zero;
    int8_t logit_one = 1;

    llama_batch batch = {};
    batch.n_tokens = 1;
    if (token_input) {
        batch.token = &token;
        batch.embd = nullptr;
    } else {
        batch.token = nullptr;
        batch.embd = scratch_embd.data();
    }
    batch.pos = scratch_pos.data();
    batch.n_seq_id = &n_seq_one;
    batch.seq_id = &seq_ptr;
    batch.logits = &logit_one;

    const int decode_rc = llama_decode(state->ctx, batch);
    if (decode_rc != 0) {
        error = "benchmark llama_decode failed with code " + std::to_string(decode_rc);
        return false;
    }
    return true;
}

// ABI v6: warmup benchmark (see header contract). Runs on scratch sequence 0,
// clears that sequence's KV afterwards, and never touches slot epochs or
// sampler chains.
int32_t tr_stage_executor_benchmark(
        void * handle,
        uint32_t steps,
        tr_stage_executor_benchmark_result * out) {
    auto * state = as_state(handle);
    if (state == nullptr || out == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    out->total_ms = 0.0;
    out->steps_executed = 0;
    out->layers_executed = 0;
    if (steps == 0) {
        return set_error(state, "benchmark steps must be greater than zero", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    const uint32_t capped_steps = std::min<uint32_t>(steps, TENSORRELAY_STAGE_EXECUTOR_BENCHMARK_MAX_STEPS);

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->loaded) {
        return set_error(state, "stage executor is not loaded", TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED);
    }
    if (state->model == nullptr || state->ctx == nullptr) {
        return set_error(
            state,
            "benchmark unsupported: stage was loaded metadata-only without an executable llama runtime",
            TENSORRELAY_STAGE_EXECUTOR_ERR_UNSUPPORTED);
    }
    const uint32_t layers = state->last_layer_exclusive > state->first_layer
        ? state->last_layer_exclusive - state->first_layer
        : state->block_count;
    if (layers == 0) {
        return set_error(state, "benchmark cannot determine the stage layer count", TENSORRELAY_STAGE_EXECUTOR_ERR_UNSUPPORTED);
    }

    std::vector<float> scratch_embd;
    if (state->split_count > 1 && state->stage_index > 0) {
        scratch_embd.assign(state->hidden_dim, 0.0f);
    }
    std::vector<llama_pos> scratch_pos;
    std::string error;

    // Untimed priming step: first-run graph build/allocation must not skew
    // the per-layer measurement.
    if (!benchmark_decode_step(state, scratch_embd, scratch_pos, 0, error)) {
        llama_memory_seq_rm(llama_get_memory(state->ctx), 0, -1, -1);
        return set_error(state, error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }

    const auto started = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < capped_steps; ++i) {
        if (!benchmark_decode_step(state, scratch_embd, scratch_pos, static_cast<llama_pos>(i + 1), error)) {
            llama_memory_seq_rm(llama_get_memory(state->ctx), 0, -1, -1);
            return set_error(state, error, TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
        }
    }
    const auto finished = std::chrono::steady_clock::now();

    // Clear the scratch sequence so benchmark KV never leaks into a slot.
    llama_memory_seq_rm(llama_get_memory(state->ctx), 0, -1, -1);

    out->total_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    out->steps_executed = capped_steps;
    out->layers_executed = layers;
    state->last_error.clear();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

// ABI v7: enumerate the ggml GPU/iGPU devices loadable in this process right
// now (see header contract). Requires only a created handle, not a loaded
// stage, so a supervisor can probe real device availability up front.
int32_t tr_stage_executor_enumerate_devices(
        void * handle,
        uint8_t * out_ptr,
        size_t out_cap,
        size_t * out_len) {
    auto * state = as_state(handle);
    if (state == nullptr || out_len == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    *out_len = 0;
    std::lock_guard<std::mutex> lock(state->mutex);
    std::call_once(backend_init_once, []() {
        llama_backend_init();
        ggml_backend_load_all();
    });

    json devices = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * name = ggml_backend_dev_name(dev);
        const char * backend_name = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
        const char * description = ggml_backend_dev_description(dev);
        const uint64_t max_mb = std::numeric_limits<uint32_t>::max();
        devices.push_back(json{
            { "name", name == nullptr ? "" : name },
            { "backend", backend_name == nullptr ? "" : backend_name },
            { "description", description == nullptr ? "" : description },
            { "device_type", type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "igpu" : "gpu" },
            { "memory_total_mb", std::min<uint64_t>(total_bytes / (1024u * 1024u), max_mb) },
            { "memory_free_mb", std::min<uint64_t>(free_bytes / (1024u * 1024u), max_mb) },
        });
    }
    const std::string payload = devices.dump();
    *out_len = payload.size();
    if (out_ptr == nullptr || out_cap == 0) {
        // probe call: report the required capacity only
        state->last_error.clear();
        return TENSORRELAY_STAGE_EXECUTOR_OK;
    }
    if (out_cap < payload.size()) {
        return set_error(state, "device enumeration buffer is too small", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    std::memcpy(out_ptr, payload.data(), payload.size());
    state->last_error.clear();
    return TENSORRELAY_STAGE_EXECUTOR_OK;
}

// ABI v5: install the per-(slot, request epoch) sampler chain used by
// final-stage sampling. Reservation-time call; replaces any previous chain.
int32_t tr_stage_executor_configure_slot_sampler(
        void * handle,
        const tr_stage_executor_sampler_params * params) {
    auto * state = as_state(handle);
    if (state == nullptr || params == nullptr) {
        return TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT;
    }
    if (params->abi_version != TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION) {
        return set_error(state, "stage executor sampler ABI version mismatch", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->loaded || state->slot_samplers.empty()) {
        return set_error(state, "stage executor slots are not allocated", TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED);
    }
    if (params->slot_id >= state->slot_samplers.size()) {
        return set_error(state, "sampler configuration references an unknown slot", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (params->request_epoch < state->slot_epochs[params->slot_id]) {
        return set_error(state, "sampler configuration references a stale slot epoch", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    if (!std::isfinite(params->temperature) || !std::isfinite(params->top_p) || !std::isfinite(params->min_p)) {
        return set_error(state, "sampler configuration contains non-finite values", TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT);
    }
    auto & slot = state->slot_samplers[params->slot_id];
    reset_slot_sampler(slot);
    slot.sampler = create_sampler_from_params(*params);
    slot.request_epoch = params->request_epoch;
    slot.configured = true;
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
