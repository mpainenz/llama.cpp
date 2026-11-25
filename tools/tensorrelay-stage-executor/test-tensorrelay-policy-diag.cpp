// Decisions stay the caller's values. The record stores the first gate's
// metadata and drops anything that is not an identifier.

#include "ggml-tensorrelay-policy-diag.h"

#include <cstdio>
#include <cstring>
#include <string>

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

static int existing_decision(int supports_op) {
    return supports_op ? 0 : 4;
}

int main() {
    const char * src_types[] = {"f32", "i32"};
    tr_policy_diag_reset();

    const int allow = tr_policy_diag_preserve(
        existing_decision(1), 0,
        "supports_op", "SET_ROWS", "", src_types, 2, "f32", "Vulkan", "Vulkan0", "supports_op", 1);
    expect(allow == 0, "supported op is still allowed");
    tr_policy_diag_record empty;
    expect(tr_policy_diag_copy(&empty) == 0, "an allow does not record a refusal");

    const int deny = tr_policy_diag_preserve(
        existing_decision(0), 1,
        "supports_op", "SET_ROWS", "type_mismatch", src_types, 2, "f32", "Vulkan", "Vulkan0", "4.assign", 0);
    expect(deny == 4, "unsupported op is still refused with code 4");

    const int deny_again = tr_policy_diag_preserve(
        existing_decision(0), 1,
        "scheduler", "MUL_MAT", "", src_types, 2, "f16", "CPU", "CPU", "5.split", 0);
    expect(deny_again == 4, "a later refusal keeps the same decision");

    tr_policy_diag_record first;
    expect(tr_policy_diag_copy(&first) == 1, "the first refusal is recorded");
    expect(std::strcmp(first.gate, "supports_op") == 0, "gate is the first failing check");
    expect(std::strcmp(first.op, "SET_ROWS") == 0, "op is recorded");
    expect(std::strcmp(first.subtype, "type_mismatch") == 0, "subtype is recorded");
    expect(first.n_src == 2, "source type count is recorded");
    expect(std::strcmp(first.src_type[0], "f32") == 0, "first source type is recorded");
    expect(std::strcmp(first.src_type[1], "i32") == 0, "second source type is recorded");
    expect(std::strcmp(first.dst_type, "f32") == 0, "destination type is recorded");
    expect(std::strcmp(first.backend, "Vulkan") == 0, "backend is recorded");
    expect(std::strcmp(first.device, "Vulkan0") == 0, "selected device is recorded");
    expect(std::strcmp(first.phase, "4.assign") == 0, "scheduler phase is recorded");
    expect(first.supports_op == 0, "supports_op is recorded");
    expect(std::strcmp(first.gate, "scheduler") != 0, "a later gate does not replace the first");

    char line[1024];
    const size_t n = tr_policy_diag_format(line, sizeof line);
    expect(n > 0 && std::strlen(line) == n, "format length matches the line");
    const std::string text(line);
    expect(text.find("gate=supports_op") != std::string::npos, "line names the gate");
    expect(text.find("op=SET_ROWS") != std::string::npos, "line names the op");
    expect(text.find("subtype=type_mismatch") != std::string::npos, "line names the subtype");
    expect(text.find("src0=f32") != std::string::npos, "line names the first type");
    expect(text.find("src1=i32") != std::string::npos, "line names the second type");
    expect(text.find("dst=f32") != std::string::npos, "line names the destination type");
    expect(text.find("backend=Vulkan") != std::string::npos, "line names the backend");
    expect(text.find("device=Vulkan0") != std::string::npos, "line names the device");
    expect(text.find("phase=4.assign") != std::string::npos, "line names the phase");
    expect(text.find("supports_op=0") != std::string::npos, "line names supports_op");
    expect(text.find("contents=") == std::string::npos, "line has no tensor contents field");
    expect(text.find("token=") == std::string::npos, "line has no token field");
    expect(text.find("credential=") == std::string::npos, "line has no credential field");
    expect(text.find("CPU") == std::string::npos, "the second refusal is absent");

    unsigned char payload[200];
    std::memset(payload, 0xAB, sizeof payload);
    payload[sizeof payload - 1] = 0;
    const char * leaked[] = {reinterpret_cast<const char *>(payload)};
    tr_policy_diag_reset();
    tr_policy_diag_note(
        "supports_op", "SET_ROWS", "credential=supersecret",
        leaked, 1, "f32",
        "token=supersecret",
        reinterpret_cast<const char *>(payload),
        "1.buffer", 0);
    char sanitized[1024];
    tr_policy_diag_format(sanitized, sizeof sanitized);
    expect(std::memchr(sanitized, 0xAB, std::strlen(sanitized)) == nullptr, "raw tensor bytes are not copied");
    expect(std::strstr(sanitized, "token=") == nullptr, "a token assignment is not a field");
    expect(std::strstr(sanitized, "credential=") == nullptr, "a credential assignment is not a field");
    expect(std::strstr(sanitized, "src0=") != nullptr, "the sanitized source type is still a type field");

    tr_policy_diag_record stored;
    tr_policy_diag_copy(&stored);
    expect(std::strlen(stored.backend) < TR_POLICY_DIAG_IDENT, "backend identifier is bounded");
    expect(std::strlen(stored.device) < TR_POLICY_DIAG_IDENT, "device identifier is bounded");
    bool only_ident = true;
    for (const char * p = sanitized; *p != '\0'; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '.' || c == '-' || c == ':' || c == '=' || c == ' ')) {
            only_ident = false;
        }
    }
    expect(only_ident, "the line contains only metadata characters");

    std::printf("%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
