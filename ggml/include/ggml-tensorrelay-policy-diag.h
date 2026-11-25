// Diagnostic-only record of the first TensorRelay policy-node refusal.
//
// Callers keep their existing allow/deny decision. This store does not
// consult an allowlist and does not change a return code. It keeps one
// record: the first failing gate, the op and subtype, tensor types, the
// backend and selected device, the scheduler phase, and supports_op.
// There is no field for tensor bytes, tokens, or credentials. Identifiers
// are copied into fixed buffers and characters outside a name alphabet
// are replaced, so a payload cannot land in the line as itself.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TR_POLICY_DIAG_IDENT 64
#define TR_POLICY_DIAG_SRC 10

typedef struct tr_policy_diag_record {
    char gate[TR_POLICY_DIAG_IDENT];
    char op[TR_POLICY_DIAG_IDENT];
    char subtype[TR_POLICY_DIAG_IDENT];
    char src_type[TR_POLICY_DIAG_SRC][TR_POLICY_DIAG_IDENT];
    int n_src;
    char dst_type[TR_POLICY_DIAG_IDENT];
    char backend[TR_POLICY_DIAG_IDENT];
    char device[TR_POLICY_DIAG_IDENT];
    char phase[TR_POLICY_DIAG_IDENT];
    int supports_op;
    int recorded;
} tr_policy_diag_record;

// Stores the note only when no record exists yet. Returns 1 if this call
// stored the first record, 0 if a record was already present or the gate
// name is empty. supports_op is clamped to -1, 0, or 1.
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
    int supports_op);

void tr_policy_diag_reset(void);

// Copies the current record. Returns 1 when a record is present.
int tr_policy_diag_copy(tr_policy_diag_record * out);

// Writes one metadata line, or nothing when no record exists.
// Returns the number of bytes that would be written, excluding the NUL.
size_t tr_policy_diag_format(char * out, size_t cap);

// Returns `decision` unchanged. When `failing` is non-zero, notes the
// gate first. The note cannot replace the decision.
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
    int supports_op);

#ifdef __cplusplus
}
#endif
