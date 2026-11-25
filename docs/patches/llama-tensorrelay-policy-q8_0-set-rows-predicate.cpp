// Source-independent mirror of policy_state::guard SET_ROWS.
// Does not include the LAN policy file. Device and supports_op stay after this check.
// Q8_0 codes are not bit-exact with the CPU reference.

#include <stdint.h>
#include <stdio.h>

enum { TYPE_F32 = 0, TYPE_F16 = 1, TYPE_Q4_0 = 2, TYPE_Q4_1 = 3, TYPE_Q5_0 = 6, TYPE_Q8_0 = 8, TYPE_I32 = 26, TYPE_I64 = 27 };
enum { QK8_0 = 32 };

// true = the case falls through. false = policy.reject and return false.
static bool set_rows_allowed(bool has_src0, bool has_src1, int src0, int src1, int dst, int64_t ne0) {
    if (!has_src0 || !has_src1 || src0 != TYPE_F32 || src1 != TYPE_I64 ||
            !(dst == TYPE_F16 || (dst == TYPE_Q8_0 && ne0 % QK8_0 == 0))) {
        return false;
    }
    return true;
}

static int g_failed;

static void expect(bool got, bool want, const char * name) {
    if (got == want) {
        printf("ok %s\n", name);
        return;
    }
    printf("FAIL %s\n", name);
    g_failed++;
}

int main() {
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_F16, 512), true, "F16 allowed");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_F16, 513), true, "F16 ignores Q8_0 alignment");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_Q8_0, 512), true, "Q8_0 aligned allowed");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_Q8_0, 65536), true, "Q8_0 width 65536 allowed");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_Q8_0, 513), false, "Q8_0 misaligned refused");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_Q8_0, 16), false, "Q8_0 width 16 refused");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_Q4_0, 512), false, "Q4_0 destination refused");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_Q5_0, 512), false, "Q5_0 destination refused");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_Q4_1, 512), false, "Q4_1 destination refused");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I64, TYPE_F32, 512), false, "F32 destination refused");
    expect(set_rows_allowed(true, true, TYPE_F16, TYPE_I64, TYPE_Q8_0, 512), false, "non-F32 src0 refused");
    expect(set_rows_allowed(true, true, TYPE_F32, TYPE_I32, TYPE_Q8_0, 512), false, "non-I64 src1 refused");
    expect(set_rows_allowed(false, true, TYPE_F32, TYPE_I64, TYPE_F16, 512), false, "null src0 refused");
    expect(set_rows_allowed(true, false, TYPE_F32, TYPE_I64, TYPE_F16, 512), false, "null src1 refused");
    printf("%d failed\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
