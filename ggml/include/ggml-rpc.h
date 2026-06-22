#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
#include <unordered_map>
#include <string>

struct TensorLocation {
    std::string path;
    size_t offset;
    size_t size;
};
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    3
#define RPC_PROTO_MINOR_VERSION    6
#define RPC_PROTO_PATCH_VERSION    1

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 96, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

// TensorRelay model-part-stub fast path: bind a tensor that lives on a remote RPC
// buffer to data the remote already holds, identified by a precomputed FNV-1a hash,
// without transferring any bytes. Returns true iff the remote resolved the hash
// (from its local GGUF tensor map or file cache) and set the tensor; false on a
// cache miss, a non-RPC buffer, or a transport error. Stage-0 uses this to "load"
// the stripped (> HASH_THRESHOLD) peer tensors it never downloaded: the hash comes
// from the .stub.gguf and the peer that owns the shard resolves the bytes locally.
GGML_BACKEND_API bool ggml_backend_rpc_buffer_set_tensor_hash(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint64_t hash, size_t offset);

#ifdef __cplusplus
}
GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices,
                                                    const std::unordered_map<uint64_t, TensorLocation> * tensor_map_ptr = nullptr);
extern "C" {
#else
GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices,
                                                    const void * tensor_map_ptr);
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
