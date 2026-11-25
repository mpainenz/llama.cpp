// Scoped SET_ROWS gate for the TensorRelay worker.
//
// The existing allow is an F16 destination. One more tuple is allowed:
// F32 values, I64 row indexes, Q8_0 destination, on the assigned Vulkan
// device, and only when that device reports supports_op and the pipeline
// is present. CPU assignment and every other type or shape stay refused.
// This does not copy tensor bytes.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TR_POLICY_OP_SET_ROWS 42
#define TR_POLICY_TYPE_F32 0
#define TR_POLICY_TYPE_F16 1
#define TR_POLICY_TYPE_Q4_0 2
#define TR_POLICY_TYPE_Q8_0 8
#define TR_POLICY_TYPE_I32 26
#define TR_POLICY_TYPE_I64 27
#define TR_POLICY_Q8_0_BLOCK 32
#define TR_POLICY_NODE_REFUSED 4

// reason 1: destination or source type is outside the scoped allow
// reason 2: device is not the assigned Vulkan device
// reason 3: shape or Q8_0 block alignment
// reason 4: supports_op is false
// reason 5: the Vulkan pipeline is absent
#define TR_POLICY_REASON_TYPE 1
#define TR_POLICY_REASON_DEVICE 2
#define TR_POLICY_REASON_SHAPE 3
#define TR_POLICY_REASON_SUPPORTS_OP 4
#define TR_POLICY_REASON_PIPELINE 5

typedef struct tr_set_rows_node {
    int op;
    int dst_type;
    int src0_type;
    int src1_type;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t ne3;
    int assigned_vulkan;
    int cpu;
    int supports_op;
    int pipeline_present;
} tr_set_rows_node;

// 0 keeps the node. TR_POLICY_NODE_REFUSED refuses it.
// *reason is set on a refusal and cleared on an allow. reason may be null.
int tr_policy_set_rows_decide(const tr_set_rows_node * node, int * reason);

// Writes quantized rows. quant_row must write exactly row_bytes.
// Selected rows are replaced. Every other row is left as it was.
// Returns 0, or TR_POLICY_NODE_REFUSED when the indexes are out of range.
int tr_policy_set_rows_write(
    void * dst,
    size_t row_bytes,
    int64_t n_rows,
    const float * values,
    int64_t n_cols,
    const int64_t * rows,
    int n_rows_selected,
    void (*quant_row)(const float * src, void * dst, int64_t n_cols, void * user),
    void * user);

// The worker arms the assigned Vulkan device names before model load.
// An empty list arms the gate with no Vulkan device, so the Q8_0 tuple
// cannot fall through to CPU. F16 destinations stay allowed.
void tr_policy_set_rows_arm(const char * const * device_names, int count);
void tr_policy_set_rows_disarm(void);
int tr_policy_set_rows_armed(void);
int tr_policy_set_rows_assigned(const char * backend_name);

#ifdef __cplusplus
}
#endif
