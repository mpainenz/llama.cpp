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

#define TENSORRELAY_STAGE_EXECUTOR_ABI_VERSION 4u
#define TENSORRELAY_STAGE_EXECUTOR_OK 0
#define TENSORRELAY_STAGE_EXECUTOR_ERR_INVALID_ARGUMENT -1
#define TENSORRELAY_STAGE_EXECUTOR_ERR_NOT_LOADED -2
#define TENSORRELAY_STAGE_EXECUTOR_ERR_UNSUPPORTED -40

#define TENSORRELAY_STAGE_EXECUTOR_INPUT_FLAG_REQUEST_JSON (1u << 0)
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
// ABI v4: applies the loaded model chat template to an OpenAI-style request JSON payload
// and tokenizes the result. out_token_count is always set to the required token count on
// success; token ids are written only when out_tokens_ptr is non-NULL and out_tokens_cap
// >= *out_token_count. Callers use the two-call pattern (probe with cap 0, then fill).
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_tokenize(
    void * handle,
    const uint8_t * request_json_ptr,
    size_t request_json_len,
    int32_t * out_tokens_ptr,
    size_t out_tokens_cap,
    size_t * out_token_count);
TENSORRELAY_STAGE_EXECUTOR_API int32_t  tr_stage_executor_release_slot(void * handle, uint32_t slot_id, uint64_t request_epoch);
TENSORRELAY_STAGE_EXECUTOR_API size_t   tr_stage_executor_last_error(void * handle, uint8_t * out_ptr, size_t out_len);

#ifdef __cplusplus
}
#endif
