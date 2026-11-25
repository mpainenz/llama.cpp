#include "ggml-tensorrelay-policy-diag.h"

#include <mutex>
#include <stdio.h>
#include <string.h>

static std::mutex g_policy_diag_mu;
static tr_policy_diag_record g_policy_diag;

static void tr_policy_diag_copy_ident(char * dst, size_t cap, const char * src) {
    size_t n = 0;
    if (cap == 0) {
        return;
    }
    if (src != nullptr) {
        for (; src[n] != '\0' && n + 1 < cap; ++n) {
            const unsigned char c = static_cast<unsigned char>(src[n]);
            const bool ok =
                (c >= 'A' && c <= 'Z') ||
                (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9') ||
                c == '_' || c == '.' || c == '-' || c == ':';
            dst[n] = ok ? static_cast<char>(c) : '_';
        }
    }
    dst[n] = '\0';
}

int tr_policy_diag_note(
        const char * gate,
        const char * op,
        const char * subtype,
        const char * const * src_types,
        int n_src,
        const char * dst_type,
        const char * backend,
        const char * device,
        const char * phase,
        int supports_op) {
    if (gate == nullptr || gate[0] == '\0') {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_policy_diag_mu);
    if (g_policy_diag.recorded) {
        return 0;
    }
    tr_policy_diag_record next = {};
    tr_policy_diag_copy_ident(next.gate, sizeof next.gate, gate);
    if (next.gate[0] == '\0') {
        return 0;
    }
    tr_policy_diag_copy_ident(next.op, sizeof next.op, op);
    tr_policy_diag_copy_ident(next.subtype, sizeof next.subtype, subtype);
    if (n_src < 0) {
        n_src = 0;
    }
    if (n_src > TR_POLICY_DIAG_SRC) {
        n_src = TR_POLICY_DIAG_SRC;
    }
    next.n_src = n_src;
    for (int i = 0; i < n_src; ++i) {
        const char * type_name = src_types != nullptr ? src_types[i] : nullptr;
        tr_policy_diag_copy_ident(next.src_type[i], sizeof next.src_type[i], type_name);
    }
    tr_policy_diag_copy_ident(next.dst_type, sizeof next.dst_type, dst_type);
    tr_policy_diag_copy_ident(next.backend, sizeof next.backend, backend);
    tr_policy_diag_copy_ident(next.device, sizeof next.device, device);
    tr_policy_diag_copy_ident(next.phase, sizeof next.phase, phase);
    if (supports_op < -1) {
        supports_op = -1;
    } else if (supports_op > 1) {
        supports_op = 1;
    }
    next.supports_op = supports_op;
    next.recorded = 1;
    g_policy_diag = next;
    return 1;
}

void tr_policy_diag_reset(void) {
    std::lock_guard<std::mutex> lock(g_policy_diag_mu);
    g_policy_diag = {};
}

int tr_policy_diag_copy(tr_policy_diag_record * out) {
    if (out == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_policy_diag_mu);
    *out = g_policy_diag;
    return g_policy_diag.recorded;
}

size_t tr_policy_diag_format(char * out, size_t cap) {
    tr_policy_diag_record rec;
    if (!tr_policy_diag_copy(&rec) || !rec.recorded) {
        if (out != nullptr && cap > 0) {
            out[0] = '\0';
        }
        return 0;
    }
    char line[1024];
    int n = snprintf(
        line, sizeof line,
        "tr_policy_diag gate=%s op=%s subtype=%s dst=%s backend=%s device=%s phase=%s supports_op=%d n_src=%d",
        rec.gate, rec.op, rec.subtype, rec.dst_type, rec.backend, rec.device, rec.phase, rec.supports_op, rec.n_src);
    if (n < 0) {
        n = 0;
    }
    for (int i = 0; i < rec.n_src && n >= 0 && static_cast<size_t>(n) + 1 < sizeof line; ++i) {
        const int wrote = snprintf(
            line + n, sizeof line - static_cast<size_t>(n),
            " src%d=%s", i, rec.src_type[i]);
        if (wrote < 0) {
            break;
        }
        n += wrote;
    }
    if (n < 0) {
        n = 0;
    }
    if (static_cast<size_t>(n) >= sizeof line) {
        n = static_cast<int>(sizeof line - 1);
    }
    line[n] = '\0';
    if (out != nullptr && cap > 0) {
        const size_t copy_n = static_cast<size_t>(n) < cap - 1 ? static_cast<size_t>(n) : cap - 1;
        memcpy(out, line, copy_n);
        out[copy_n] = '\0';
    }
    return static_cast<size_t>(n);
}

int tr_policy_diag_preserve(
        int decision,
        int failing,
        const char * gate,
        const char * op,
        const char * subtype,
        const char * const * src_types,
        int n_src,
        const char * dst_type,
        const char * backend,
        const char * device,
        const char * phase,
        int supports_op) {
    if (failing) {
        tr_policy_diag_note(
            gate, op, subtype, src_types, n_src, dst_type, backend, device, phase, supports_op);
    }
    return decision;
}
