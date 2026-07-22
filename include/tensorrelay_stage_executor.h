#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#    if defined(TENSORRELAY_STAGE_EXECUTOR_BUILD)
#        define TENSORRELAY_STAGE_EXECUTOR_API __declspec(dllexport)
#    else
#        define TENSORRELAY_STAGE_EXECUTOR_API __declspec(dllimport)
#    endif
#else
#    define TENSORRELAY_STAGE_EXECUTOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION 11u
#define TENSORRELAY_STAGE_EXECUTOR_OK 0
#define TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT -1
#define TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED -2
#define TENSORRELAY_STAGE_EXECUTOR_ERR_UNSUPPORTED -40

// ABI v6: hard cap on tr_stage_executor_benchmark decode steps.
#define TENSORRELAY_STAGE_EXECUTOR_BENCHMARK_MAX_STEPS 16u

// (1u << 0) was INPUT_FLAG_REQUEST_JSON: the single-stage one-shot blob path,
// retired in ABI v10. Single-stage execution now uses the same token loop as
// split stages (token ids in, one sampled token out per call). The bit stays
// reserved so old callers fail the version check instead of aliasing a flag.
#define TENSORRELAY_STAGE_EXECUTOR_INPUT_FLAG_FINAL_CHUNK  (1u << 1)
#define TENSORRELAY_STAGE_EXECUTOR_INPUT_FLAG_TOKEN_IDS    (1u << 2)
#define TENSORRELAY_STAGE_EXECUTOR_INPUT_FLAG_ACTIVATIONS  (1u << 3)
#define TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_FINAL_TEXT  (1u << 0)
#define TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_ACTIVATIONS (1u << 1)
#define TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_TOKEN_ID    (1u << 2)
// Set on final-stage sampled outputs when the sampled token is end-of-generation.
#define TENSORRELAY_STAGE_EXECUTOR_OUTPUT_FLAG_EOG         (1u << 3)

typedef struct tr_stage_executor_load_params {
    uint32_t    abi_version;
    int32_t     stage_index;
    int32_t     split_count;
    uint32_t    slot_count;
    uint32_t    context_length;
    const uint8_t * shard_path_ptr;
    size_t      shard_path_len;
    const uint8_t * selected_devices_ptr;
    size_t      selected_devices_len;
    // ABI v8: tail-spill (ADR-0018 Layer Spill). The last spill_layer_count
    // blocks of this stage - and the output head when this is the final stage -
    // are placed on spill_devices instead of the discrete selected_devices.
    // spill_layer_count 0 preserves v7 behavior (nothing spilled). An empty
    // spill_devices string selects the CPU (system-RAM) buffer type; otherwise
    // it names a single ggml device (e.g. an iGPU) whose buffer type receives
    // the spilled tensors. A nonzero request is applied exactly or the load
    // fails with ERR_INVALID_ARGUMENT: metadata-only or CPU-only selections,
    // a missing/invalid stage layer range, a count above the stage's span, an
    // unknown spill device, a GPU spill with multiple selected head devices,
    // or a spill device that is already selected are all errors. Capabilities
    // echo the applied spill (spill_layer_count, spill_devices). Notes: KV
    // cache for spilled blocks stays on the head device (llama places KV by
    // layer-split assignment, not weight overrides) - capacity accounting
    // must charge spilled-block KV to the head; tied-embedding models have no
    // output.* tensors, so the final-stage head remains with token_embd on
    // the CPU input device (already system memory).
    uint32_t    spill_layer_count;
    const uint8_t * spill_devices_ptr;
    size_t      spill_devices_len;
    // ABI v9: KV Cache Policy (per-model, stamped in the layer manifest).
    // cache_type_k / cache_type_v name ggml types for the KV cache ("f16",
    // "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"). Empty selects
    // f16 (the pre-v9 behavior). Requesting any non-f16 type also enables
    // flash attention (quantized caches require it); if the context cannot be
    // created with the requested types plus flash attention the load FAILS -
    // never a silent f16 fallback, which would allocate a larger cache than
    // the layer solver budgeted and OOM exactly the tight fits the budget
    // exists to protect. Capabilities echo the applied types (cache_type_k,
    // cache_type_v; empty on metadata-only loads, which create no cache) so
    // the runtime can verify requested == applied at warmup.
    const uint8_t * cache_type_k_ptr;
    size_t      cache_type_k_len;
    const uint8_t * cache_type_v_ptr;
    size_t      cache_type_v_len;
} tr_stage_executor_load_params;

typedef struct tr_stage_executor_slot_input {
    uint32_t slot_id;
    uint64_t request_epoch;
    uint32_t token_offset;
    uint32_t token_count;
    uint32_t activation_offset;
    uint32_t activation_len;
    uint32_t position_start;
    uint32_t flags;
} tr_stage_executor_slot_input;

typedef struct tr_stage_executor_slot_output {
    uint32_t slot_id;
    uint64_t request_epoch;
    uint32_t activation_offset;
    uint32_t activation_len;
    int32_t  sampled_token_id;
    uint32_t flags;
} tr_stage_executor_slot_output;

// ABI v5: per-(slot, request epoch) sampler configuration. The runtime calls
// tr_stage_executor_configure_slot_sampler at reservation time on the final
// stage; sampling steps for a matching (slot_id, request_epoch) then use this
// chain instead of the built-in default (min_p 0.05, temp 0.8, random seed).
// temperature <= 0 selects fully deterministic greedy decoding. top_k <= 0,
// top_p >= 1, and min_p <= 0 disable the respective samplers. seed 0xFFFFFFFF
// (LLAMA_DEFAULT_SEED) draws a random seed.
typedef struct tr_stage_executor_sampler_params {
    uint32_t abi_version;
    uint32_t slot_id;
    uint64_t request_epoch;
    float    temperature;
    float    top_p;
    float    min_p;
    int32_t  top_k;
    uint32_t seed;
    uint32_t reserved;
} tr_stage_executor_sampler_params;

// ABI v6: warmup benchmark result. total_ms is the wall-clock time across the
// timed decode steps; ms-per-layer = total_ms / (steps_executed * layers_executed).
typedef struct tr_stage_executor_benchmark_result {
    double   total_ms;
    uint32_t steps_executed;
    uint32_t layers_executed;
} tr_stage_executor_benchmark_result;

typedef struct tr_stage_executor_batch_call {
    uint32_t abi_version;
    uint32_t input_count;
    const tr_stage_executor_slot_input * inputs_ptr;
    const uint8_t * input_bytes_ptr;
    size_t input_bytes_len;
    uint8_t * output_bytes_ptr;
    size_t output_bytes_len;
    tr_stage_executor_slot_output * outputs_ptr;
    uint32_t outputs_len;
} tr_stage_executor_batch_call;

TENSORRELAY_STAGE_EXECUTOR_API uint32_t tr_stage_executor_abi_version(void);
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_create(void ** out_handle);
TENSORRELAY_STAGE_EXECUTOR_API void     tr_stage_executor_destroy(void * handle);
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_load(void * handle, const tr_stage_executor_load_params * params);
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_alloc_slots(void * handle, uint32_t slot_count);
TENSORRELAY_STAGE_EXECUTOR_API size_t   tr_stage_executor_activation_descriptor_json(
    void * handle,
    int32_t source_stage,
    int32_t target_stage,
    uint8_t * out_ptr,
    size_t out_len);
TENSORRELAY_STAGE_EXECUTOR_API size_t   tr_stage_executor_capabilities_json(
    void * handle,
    uint8_t * out_ptr,
    size_t out_len);
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_execute_batch(void * handle, tr_stage_executor_batch_call * call);
// ABI v10: renders the loaded model chat template for an OpenAI chat-completions
// request JSON (messages[], optional tools[]/tool_choice/parallel_tool_calls) and
// tokenizes the result. Replaces the v4 tr_stage_executor_tokenize entry, which
// rendered through the tools-blind legacy template path.
//
// Two outputs, both two-call pattern: on success *out_token_count and
// *out_state_len are always set to the required sizes; token ids and the state
// JSON are written only when the matching buffer is present and large enough.
// The state JSON is the Chat Format Handle: an opaque blob the caller must pass
// back verbatim to tr_stage_executor_chat_parse for this generation. It also
// carries "additional_stops" (format-implied stop strings the caller must
// enforce; the token loop does not scan text) and "prompt_token_count".
// tool_choice may be absent, "auto", or "none" - the caller rejects "required"
// and named-function forcing before this entry is reached (no grammar
// constraining in v10).
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_chat_begin(
    void * handle,
    const uint8_t * request_json_ptr,
    size_t request_json_len,
    int32_t * out_tokens_ptr,
    size_t out_tokens_cap,
    size_t * out_token_count,
    uint8_t * out_state_ptr,
    size_t out_state_cap,
    size_t * out_state_len);
// ABI v10: parses accumulated generated text into structured chat state using
// the Chat Format Handle from tr_stage_executor_chat_begin. is_partial nonzero
// while generation is still running (the parser then tolerates and heals
// incomplete syntax); zero for the final call. Writes a JSON object
// {"content","reasoning_content","tool_calls":[{"id","name","arguments"}]}
// reflecting the ENTIRE text so far - the caller diffs successive states into
// streaming deltas. Two-call pattern via out_ptr/out_cap/out_len. Reads only
// its arguments (no llama state) but serializes on the executor mutex like
// every other entry.
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_chat_parse(
    void * handle,
    const uint8_t * state_json_ptr,
    size_t state_json_len,
    const uint8_t * text_ptr,
    size_t text_len,
    uint32_t is_partial,
    uint8_t * out_ptr,
    size_t out_cap,
    size_t * out_len);
// ABI v5: installs the sampler chain used for final-stage sampling of the
// given (slot, request epoch). Stale epochs (older than the slot's current
// epoch) are rejected. Reconfiguring replaces any previous chain for the slot.
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_configure_slot_sampler(
    void * handle,
    const tr_stage_executor_sampler_params * params);
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_release_slot(void * handle, uint32_t slot_id, uint64_t request_epoch);
// ABI v6: runs `steps` (capped at TENSORRELAY_STAGE_EXECUTOR_BENCHMARK_MAX_STEPS)
// batch-1 decode-shaped steps through the loaded stage graph on scratch
// sequence 0 and reports total wall-clock time plus the stage's layer count.
// One untimed priming step precedes the timed loop so first-run graph
// allocation does not skew the measurement. The scratch sequence's KV cells
// are cleared afterwards; slot epochs and samplers are untouched. Call during
// warmup, before any slot holds live KV state (the scratch sequence aliases
// slot 0's KV lane). Metadata-only stage loads without an executable llama
// runtime return TENSORRELAY_STAGE_EXECUTOR_ERR_UNSUPPORTED.
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_benchmark(
    void * handle,
    uint32_t steps,
    tr_stage_executor_benchmark_result * out);
// ABI v7: enumerates the ggml backend devices that are actually loadable in
// this process right now (after ggml_backend_load_all). Works on a freshly
// created executor before any load. Writes a UTF-8 JSON array of
// {name, backend, description, device_type, memory_total_mb, memory_free_mb}
// objects, one per GPU/iGPU device (CPU and accelerator devices are omitted).
// ABI v8: device_type is "gpu" (discrete) or "igpu" (integrated), read from
// the ggml device type - the client no longer has to guess integrated vs.
// discrete from the description string. Two-call pattern: with out_ptr NULL or
// out_cap 0 only *out_len is set to the required byte count; otherwise the JSON
// is copied when it fits.
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_enumerate_devices(
    void * handle,
    uint8_t * out_ptr,
    size_t out_cap,
    size_t * out_len);
TENSORRELAY_STAGE_EXECUTOR_API size_t   tr_stage_executor_last_error(void * handle, uint8_t * out_ptr, size_t out_len);

// ABI v11: process-wide ggml backend diagnostics captured since load - the
// WARN/ERROR lines ggml would otherwise write only to stderr, including the
// reason a backend (e.g. CUDA) failed to initialize. Handle-free: the log is
// global. Non-clearing, so a one-time init failure survives repeated reads.
// Two-call pattern: out_ptr NULL or out_len 0 returns the required byte count.
TENSORRELAY_STAGE_EXECUTOR_API size_t   tr_stage_executor_read_diagnostics(uint8_t * out_ptr, size_t out_len);

#ifdef __cplusplus
}
#endif
