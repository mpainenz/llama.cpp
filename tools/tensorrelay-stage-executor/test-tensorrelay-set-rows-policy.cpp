// Scoped SET_ROWS decisions and a CPU quantized-row reference.
// GPU execution of the same rows is for the Workshop replay, not this process.

#include "ggml-tensorrelay-set-rows-policy.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ggml quantize_row_q8_0_ref writes block_q8_0: fp16 scale, then int8 qs[32].
extern "C" void quantize_row_q8_0_ref(const float * x, void * y, int64_t k);

static void reference_q8_0(const float * src, void * dst, int64_t n_cols, void *) {
    quantize_row_q8_0_ref(src, dst, n_cols);
}

static int g_failed = 0;
static int g_passed = 0;

static void expect(bool cond, const char * name) {
    if (cond) {
        ++g_passed;
        std::printf("ok %s\n", name);
        return;
    }
    ++g_failed;
    std::printf("FAIL %s\n", name);
}

static tr_set_rows_node workshop_node() {
    tr_set_rows_node node = {};
    node.op = TR_POLICY_OP_SET_ROWS;
    node.src0_type = TR_POLICY_TYPE_F32;
    node.src1_type = TR_POLICY_TYPE_I64;
    node.dst_type = TR_POLICY_TYPE_Q8_0;
    node.ne0 = 512;
    node.ne1 = 65536;
    node.ne2 = 1;
    node.ne3 = 1;
    node.assigned_vulkan = 1;
    node.cpu = 0;
    node.supports_op = 1;
    node.pipeline_present = 1;
    return node;
}

static void test_decisions() {
    int reason = 99;
    const tr_set_rows_node allowed = workshop_node();
    expect(tr_policy_set_rows_decide(&allowed, &reason) == 0, "F32/I64/Q8_0 on assigned Vulkan is allowed");
    expect(reason == 0, "an allow clears the refusal reason");

    tr_set_rows_node again = allowed;
    expect(tr_policy_set_rows_decide(&again, &reason) == 0, "the same tuple stays allowed");

    tr_set_rows_node f16 = allowed;
    f16.dst_type = TR_POLICY_TYPE_F16;
    f16.supports_op = 0;
    f16.pipeline_present = 0;
    f16.assigned_vulkan = 0;
    f16.cpu = 1;
    expect(tr_policy_set_rows_decide(&f16, &reason) == 0, "existing F16 destination stays allowed");

    tr_set_rows_node f16_idx = f16;
    f16_idx.src0_type = TR_POLICY_TYPE_F16;
    f16_idx.src1_type = TR_POLICY_TYPE_I32;
    expect(tr_policy_set_rows_decide(&f16_idx, &reason) == 0, "F16 values with I32 indexes stay allowed");

    tr_set_rows_node on_cpu = allowed;
    on_cpu.assigned_vulkan = 0;
    on_cpu.cpu = 1;
    expect(tr_policy_set_rows_decide(&on_cpu, &reason) == TR_POLICY_NODE_REFUSED, "CPU assignment of Q8_0 is refused");
    expect(reason == TR_POLICY_REASON_DEVICE, "CPU refusal names the device gate");

    tr_set_rows_node other = allowed;
    other.assigned_vulkan = 0;
    other.cpu = 0;
    expect(tr_policy_set_rows_decide(&other, &reason) == TR_POLICY_NODE_REFUSED, "a different device is refused");
    expect(reason == TR_POLICY_REASON_DEVICE, "wrong device names the device gate");

    tr_set_rows_node q4 = allowed;
    q4.dst_type = TR_POLICY_TYPE_Q4_0;
    expect(tr_policy_set_rows_decide(&q4, &reason) == TR_POLICY_NODE_REFUSED, "Q4_0 destination is refused");
    expect(reason == TR_POLICY_REASON_TYPE, "Q4_0 refusal names the type gate");

    tr_set_rows_node f32_dst = allowed;
    f32_dst.dst_type = TR_POLICY_TYPE_F32;
    expect(tr_policy_set_rows_decide(&f32_dst, &reason) == TR_POLICY_NODE_REFUSED, "F32 destination is refused");

    tr_set_rows_node f16_src = allowed;
    f16_src.src0_type = TR_POLICY_TYPE_F16;
    expect(tr_policy_set_rows_decide(&f16_src, &reason) == TR_POLICY_NODE_REFUSED, "F16 values into Q8_0 are refused");

    tr_set_rows_node i32_idx = allowed;
    i32_idx.src1_type = TR_POLICY_TYPE_I32;
    expect(tr_policy_set_rows_decide(&i32_idx, &reason) == TR_POLICY_NODE_REFUSED, "I32 indexes into Q8_0 are refused");

    tr_set_rows_node misaligned = allowed;
    misaligned.ne0 = 513;
    expect(tr_policy_set_rows_decide(&misaligned, &reason) == TR_POLICY_NODE_REFUSED, "a width that is not a Q8_0 block is refused");
    expect(reason == TR_POLICY_REASON_SHAPE, "misaligned width names the shape gate");

    tr_set_rows_node short_row = allowed;
    short_row.ne0 = 16;
    expect(tr_policy_set_rows_decide(&short_row, &reason) == TR_POLICY_NODE_REFUSED, "a 16-wide Q8_0 row is refused");

    tr_set_rows_node unsupported = allowed;
    unsupported.supports_op = 0;
    expect(tr_policy_set_rows_decide(&unsupported, &reason) == TR_POLICY_NODE_REFUSED, "supports_op false is refused");
    expect(reason == TR_POLICY_REASON_SUPPORTS_OP, "missing support names supports_op");

    tr_set_rows_node no_pipe = allowed;
    no_pipe.pipeline_present = 0;
    expect(tr_policy_set_rows_decide(&no_pipe, &reason) == TR_POLICY_NODE_REFUSED, "a missing pipeline is refused");
    expect(reason == TR_POLICY_REASON_PIPELINE, "missing pipeline names the pipeline gate");
}

static void test_rows() {
    const int64_t n_cols = 32;
    const int64_t n_rows = 4;
    const size_t row_bytes = sizeof(uint16_t) + TR_POLICY_Q8_0_BLOCK;
    std::vector<unsigned char> dst(static_cast<size_t>(n_rows) * row_bytes, 0x5A);
    std::vector<float> step1(static_cast<size_t>(2 * n_cols));
    std::vector<float> step2(static_cast<size_t>(2 * n_cols));
    for (int64_t i = 0; i < n_cols; ++i) {
        step1[i] = 1.0f;
        step1[n_cols + i] = -0.5f;
        step2[i] = 0.25f;
        step2[n_cols + i] = 0.75f;
    }
    const int64_t rows1[2] = {0, 2};
    expect(tr_policy_set_rows_write(
               dst.data(), row_bytes, n_rows, step1.data(), n_cols, rows1, 2, reference_q8_0, nullptr) == 0,
        "first decode step writes");

    std::vector<unsigned char> expect0(row_bytes);
    std::vector<unsigned char> expect2(row_bytes);
    reference_q8_0(step1.data(), expect0.data(), n_cols, nullptr);
    reference_q8_0(step1.data() + n_cols, expect2.data(), n_cols, nullptr);
    expect(std::memcmp(dst.data(), expect0.data(), row_bytes) == 0, "selected row 0 matches the reference");
    expect(std::memcmp(dst.data() + 2 * row_bytes, expect2.data(), row_bytes) == 0, "selected row 2 matches the reference");
    bool row1_untouched = true;
    bool row3_untouched = true;
    for (size_t i = 0; i < row_bytes; ++i) {
        row1_untouched = row1_untouched && dst[row_bytes + i] == 0x5A;
        row3_untouched = row3_untouched && dst[3 * row_bytes + i] == 0x5A;
    }
    expect(row1_untouched, "unselected row 1 is unchanged after step 1");
    expect(row3_untouched, "unselected row 3 is unchanged after step 1");
    bool qs_full = true;
    for (int j = 0; j < TR_POLICY_Q8_0_BLOCK; ++j) {
        qs_full = qs_full && static_cast<int8_t>(expect0[sizeof(uint16_t) + j]) == 127;
    }
    expect(qs_full, "a constant 1 row quantizes to Q8_0 magnitude 127");

    const int64_t rows2[2] = {2, 1};
    expect(tr_policy_set_rows_write(
               dst.data(), row_bytes, n_rows, step2.data(), n_cols, rows2, 2, reference_q8_0, nullptr) == 0,
        "second decode step writes");
    expect(std::memcmp(dst.data(), expect0.data(), row_bytes) == 0, "row 0 from the first step stays put");
    std::vector<unsigned char> expect2b(row_bytes);
    std::vector<unsigned char> expect1(row_bytes);
    reference_q8_0(step2.data(), expect2b.data(), n_cols, nullptr);
    reference_q8_0(step2.data() + n_cols, expect1.data(), n_cols, nullptr);
    expect(std::memcmp(dst.data() + 2 * row_bytes, expect2b.data(), row_bytes) == 0, "row 2 matches the second step");
    expect(std::memcmp(dst.data() + row_bytes, expect1.data(), row_bytes) == 0, "row 1 matches the second step");
    row3_untouched = true;
    for (size_t i = 0; i < row_bytes; ++i) {
        row3_untouched = row3_untouched && dst[3 * row_bytes + i] == 0x5A;
    }
    expect(row3_untouched, "row 3 stays untouched across both steps");

    std::vector<float> step3(static_cast<size_t>(n_cols), -1.0f);
    const int64_t rows3[1] = {0};
    expect(tr_policy_set_rows_write(
               dst.data(), row_bytes, n_rows, step3.data(), n_cols, rows3, 1, reference_q8_0, nullptr) == 0,
        "third decode step writes");
    std::vector<unsigned char> expect0b(row_bytes);
    reference_q8_0(step3.data(), expect0b.data(), n_cols, nullptr);
    expect(std::memcmp(dst.data(), expect0b.data(), row_bytes) == 0, "row 0 matches the third step");
    expect(std::memcmp(dst.data() + row_bytes, expect1.data(), row_bytes) == 0, "row 1 stays at the second step");
    expect(std::memcmp(dst.data() + 2 * row_bytes, expect2b.data(), row_bytes) == 0, "row 2 stays at the second step");
    row3_untouched = true;
    for (size_t i = 0; i < row_bytes; ++i) {
        row3_untouched = row3_untouched && dst[3 * row_bytes + i] == 0x5A;
    }
    expect(row3_untouched, "row 3 stays untouched across three steps");

    std::vector<unsigned char> frozen = dst;
    const int64_t bad[1] = {9};
    expect(tr_policy_set_rows_write(
               dst.data(), row_bytes, n_rows, step2.data(), n_cols, bad, 1, reference_q8_0, nullptr) == TR_POLICY_NODE_REFUSED,
        "an index outside the cache is refused");
    expect(frozen == dst, "a refused index does not change stored rows");
}

static void test_arm() {
    const char * names[] = {"Vulkan0"};
    tr_policy_set_rows_arm(names, 1);
    expect(tr_policy_set_rows_armed() == 1, "the worker gate can be armed");
    expect(tr_policy_set_rows_assigned("Vulkan0") == 1, "Vulkan0 is the assigned device");
    expect(tr_policy_set_rows_assigned("Vulkan1") == 0, "Vulkan1 is not the assigned device");
    expect(tr_policy_set_rows_assigned("CPU") == 0, "CPU is not the assigned device");
    tr_policy_set_rows_disarm();
    expect(tr_policy_set_rows_armed() == 0, "disarm turns the gate off");
    expect(tr_policy_set_rows_assigned("Vulkan0") == 0, "a disarmed gate matches no device");
}

int main() {
    test_decisions();
    test_rows();
    test_arm();
    std::printf("%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
