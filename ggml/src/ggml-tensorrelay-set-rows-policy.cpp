#include "ggml-tensorrelay-set-rows-policy.h"

#include <mutex>
#include <string.h>

static std::mutex g_arm_mu;
static char g_devices[8][64];
static int g_device_count = 0;
static int g_armed = 0;

static int shape_ok(const tr_set_rows_node * node) {
    return node->ne0 > 0 && node->ne1 > 0 && node->ne2 > 0 && node->ne3 > 0;
}

static int f16_sources(const tr_set_rows_node * node) {
    const int src_ok = node->src0_type == TR_POLICY_TYPE_F32 || node->src0_type == TR_POLICY_TYPE_F16;
    const int idx_ok = node->src1_type == TR_POLICY_TYPE_I32 || node->src1_type == TR_POLICY_TYPE_I64;
    return src_ok && idx_ok;
}

static int q8_0_tuple(const tr_set_rows_node * node) {
    return node->src0_type == TR_POLICY_TYPE_F32 &&
        node->src1_type == TR_POLICY_TYPE_I64 &&
        node->dst_type == TR_POLICY_TYPE_Q8_0;
}

int tr_policy_set_rows_decide(const tr_set_rows_node * node, int * reason) {
    if (reason != nullptr) {
        *reason = 0;
    }
    if (node == nullptr || node->op != TR_POLICY_OP_SET_ROWS) {
        if (reason != nullptr) {
            *reason = TR_POLICY_REASON_TYPE;
        }
        return TR_POLICY_NODE_REFUSED;
    }
    if (!shape_ok(node)) {
        if (reason != nullptr) {
            *reason = TR_POLICY_REASON_SHAPE;
        }
        return TR_POLICY_NODE_REFUSED;
    }
    if (node->dst_type == TR_POLICY_TYPE_F16 && f16_sources(node)) {
        return 0;
    }
    if (!q8_0_tuple(node)) {
        if (reason != nullptr) {
            *reason = TR_POLICY_REASON_TYPE;
        }
        return TR_POLICY_NODE_REFUSED;
    }
    if ((node->ne0 % TR_POLICY_Q8_0_BLOCK) != 0) {
        if (reason != nullptr) {
            *reason = TR_POLICY_REASON_SHAPE;
        }
        return TR_POLICY_NODE_REFUSED;
    }
    if (node->cpu || !node->assigned_vulkan) {
        if (reason != nullptr) {
            *reason = TR_POLICY_REASON_DEVICE;
        }
        return TR_POLICY_NODE_REFUSED;
    }
    if (!node->supports_op) {
        if (reason != nullptr) {
            *reason = TR_POLICY_REASON_SUPPORTS_OP;
        }
        return TR_POLICY_NODE_REFUSED;
    }
    if (!node->pipeline_present) {
        if (reason != nullptr) {
            *reason = TR_POLICY_REASON_PIPELINE;
        }
        return TR_POLICY_NODE_REFUSED;
    }
    return 0;
}

int tr_policy_set_rows_write(
        void * dst,
        size_t row_bytes,
        int64_t n_rows,
        const float * values,
        int64_t n_cols,
        const int64_t * rows,
        int n_rows_selected,
        void (*quant_row)(const float * src, void * dst, int64_t n_cols, void * user),
        void * user) {
    if (dst == nullptr || values == nullptr || rows == nullptr || quant_row == nullptr) {
        return TR_POLICY_NODE_REFUSED;
    }
    if (row_bytes == 0 || n_cols <= 0 || n_rows <= 0 || n_rows_selected < 0) {
        return TR_POLICY_NODE_REFUSED;
    }
    for (int i = 0; i < n_rows_selected; ++i) {
        const int64_t row = rows[i];
        if (row < 0 || row >= n_rows) {
            return TR_POLICY_NODE_REFUSED;
        }
        unsigned char * slot = static_cast<unsigned char *>(dst) + static_cast<size_t>(row) * row_bytes;
        quant_row(values + static_cast<int64_t>(i) * n_cols, slot, n_cols, user);
    }
    return 0;
}

static void copy_name(char * dst, const char * src) {
    size_t n = 0;
    if (src != nullptr) {
        for (; src[n] != '\0' && n + 1 < 64; ++n) {
            dst[n] = src[n];
        }
    }
    dst[n] = '\0';
}

void tr_policy_set_rows_arm(const char * const * device_names, int count) {
    std::lock_guard<std::mutex> lock(g_arm_mu);
    g_device_count = 0;
    if (count < 0) {
        count = 0;
    }
    if (count > 8) {
        count = 8;
    }
    for (int i = 0; i < count; ++i) {
        copy_name(g_devices[g_device_count], device_names != nullptr ? device_names[i] : nullptr);
        if (g_devices[g_device_count][0] != '\0') {
            ++g_device_count;
        }
    }
    g_armed = 1;
}

void tr_policy_set_rows_disarm(void) {
    std::lock_guard<std::mutex> lock(g_arm_mu);
    g_armed = 0;
    g_device_count = 0;
}

int tr_policy_set_rows_armed(void) {
    std::lock_guard<std::mutex> lock(g_arm_mu);
    return g_armed;
}

int tr_policy_set_rows_assigned(const char * backend_name) {
    std::lock_guard<std::mutex> lock(g_arm_mu);
    if (!g_armed || backend_name == nullptr || backend_name[0] == '\0') {
        return 0;
    }
    for (int i = 0; i < g_device_count; ++i) {
        if (strcmp(g_devices[i], backend_name) == 0) {
            return 1;
        }
    }
    return 0;
}
