// Test device: discovery succeeds, but opening a GPU context must never happen.
#include "ggml-backend-impl.h"
#include <cstddef>

static ggml_backend_dev_t cpu = nullptr;
static int init_calls = 0;
static ggml_backend_device device{};
static ggml_backend_reg registry{};

extern "C" int tensorrelay_forbidden_gpu_install(const char * backend_directory) {
    ggml_backend_load_all_from_path(backend_directory);
    cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) { return -1; }
    device.reg = &registry;
    device.iface.get_name = [](ggml_backend_dev_t) { return "ForbiddenGPU"; };
    device.iface.get_description = [](ggml_backend_dev_t) { return "CPU isolation test device"; };
    device.iface.get_memory = [](ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = size_t(16) << 30; };
    device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    device.iface.get_props = [](ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
        *props = {};
        props->name = device.iface.get_name(dev);
        props->description = device.iface.get_description(dev);
        props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
        device.iface.get_memory(dev, &props->memory_free, &props->memory_total);
    };
    device.iface.init_backend = [](ggml_backend_dev_t, const char *) -> ggml_backend_t { ++init_calls; return nullptr; };
    device.iface.get_buffer_type = [](ggml_backend_dev_t) { return ggml_backend_dev_buffer_type(cpu); };
    device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return false; };
    device.iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == ggml_backend_dev_buffer_type(cpu); };
    registry.api_version = GGML_BACKEND_API_VERSION;
    registry.iface.get_name = [](ggml_backend_reg_t) { return "ForbiddenGPU"; };
    registry.iface.get_device_count = [](ggml_backend_reg_t) -> size_t { return 1; };
    registry.iface.get_device = [](ggml_backend_reg_t, size_t) { return &device; };
    ggml_backend_register(&registry);
    return 0;
}

extern "C" int tensorrelay_forbidden_gpu_init_calls() { return init_calls; }
