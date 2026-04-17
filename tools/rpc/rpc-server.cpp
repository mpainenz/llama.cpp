#include "ggml-rpc.h"
#include "ggml.h"
#include "gguf.h"
#ifdef _WIN32
#  define NOMINMAX
#  define DIRECTORY_SEPARATOR '\\'
#  include <windows.h>
#  include <fcntl.h>
#  include <io.h>
#else
#  define DIRECTORY_SEPARATOR '/'
#  include <unistd.h>
#  include <sys/stat.h>
#endif
#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

// NOTE: this is copied from common.cpp to avoid linking with libcommon
#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

// NOTE: this is copied from common.cpp to avoid linking with libcommon
// returns true if successful, false otherwise
static bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

// NOTE: this is copied from common.cpp to avoid linking with libcommon
static std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
    defined(__OpenBSD__) || defined(__NetBSD__)
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else if (std::getenv("HOME")) {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        } else {
#if defined(__linux__)
            /* no $HOME is defined, fallback to getpwuid */
            struct passwd *pw = getpwuid(getuid());
            if ((!pw) || (!pw->pw_dir)) {
                throw std::runtime_error("Failed to find $HOME directory");
            }

            cache_directory = std::string(pw->pw_dir) + std::string("/.cache/");
#else /* defined(__linux__) */
            throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#elif defined(__EMSCRIPTEN__)
        GGML_ABORT("not implemented on this platform");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

static constexpr size_t RPC_HASH_THRESHOLD = 64 * 1024; // must match HASH_THRESHOLD in ggml-rpc.cpp

static uint64_t rpc_fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool build_tensor_map(const std::string & model_path, std::unordered_map<uint64_t, TensorLocation> & tensor_map) {
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", model_path.c_str());
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    const size_t data_offset = gguf_get_data_offset(gguf_ctx);
    int indexed = 0;
    int below_threshold = 0;

    FILE * fp = fopen(model_path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Failed to reopen GGUF file for reading: %s\n", model_path.c_str());
        gguf_free(gguf_ctx);
        return false;
    }

    for (int64_t i = 0; i < n_tensors; i++) {
        const size_t tensor_size = gguf_get_tensor_size(gguf_ctx, i);
        if (tensor_size <= RPC_HASH_THRESHOLD) {
            below_threshold++;
            continue;
        }

        const size_t tensor_offset = data_offset + gguf_get_tensor_offset(gguf_ctx, i);
        std::vector<uint8_t> buf(tensor_size);

#ifdef _WIN32
        if (_fseeki64(fp, (__int64)tensor_offset, SEEK_SET) != 0) {
#else
        if (fseeko(fp, (off_t)tensor_offset, SEEK_SET) != 0) {
#endif
            fprintf(stderr, "Failed to seek to tensor %s\n", gguf_get_tensor_name(gguf_ctx, i));
            continue;
        }
        if (fread(buf.data(), 1, tensor_size, fp) != tensor_size) {
            fprintf(stderr, "Failed to read tensor %s\n", gguf_get_tensor_name(gguf_ctx, i));
            continue;
        }

        uint64_t hash = rpc_fnv_hash(buf.data(), tensor_size);
        
        TensorLocation loc;
        loc.path = model_path;
        loc.offset = tensor_offset;
        loc.size = tensor_size;
        tensor_map[hash] = loc;
        indexed++;
    }

    fclose(fp);
    gguf_free(gguf_ctx);

    printf("Tensor map: %d/%d tensors from %s indexed (threshold %zu B, %d tensors below threshold)\n", 
           indexed, (int)n_tensors, model_path.c_str(), RPC_HASH_THRESHOLD, below_threshold);
    return true;
}

struct rpc_server_params {
    std::string              host        = "127.0.0.1";
    int                      port        = 50052;
    bool                     use_cache   = false;
    std::string              model_path;
    int                      n_threads   = std::max(1U, std::thread::hardware_concurrency()/2);
    std::vector<std::string> devices;
};

static void print_usage(int /*argc*/, char ** argv, rpc_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    fprintf(stderr, "  -t, --threads N                  number of threads for the CPU device (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -d, --device <dev1,dev2,...>     comma-separated list of devices\n");
    fprintf(stderr, "  -H, --host HOST                  host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p, --port PORT                  port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -c, --cache                      enable local file cache\n");
    fprintf(stderr, "  -m, --model PATH                 pre-warm cache from a GGUF file (implies --cache)\n");
    fprintf(stderr, "\n");
}

static bool rpc_server_params_parse(int argc, char ** argv, rpc_server_params & params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "--host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                fprintf(stderr, "error: invalid number of threads: %d\n", params.n_threads);
                return false;
            }
        } else if (arg == "-d" || arg == "--device") {
            if (++i >= argc) {
                return false;
            }
            const std::regex regex{ R"([,/]+)" };
            std::string dev_str = argv[i];
            std::sregex_token_iterator iter(dev_str.begin(), dev_str.end(), regex, -1);
            std::sregex_token_iterator end;
            for ( ; iter != end; ++iter) {
                try {
                    params.devices.push_back(*iter);
                } catch (const std::exception & ) {
                    fprintf(stderr, "error: invalid device: %s\n", iter->str().c_str());
                    return false;
                }
            }
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        } else if (arg == "-c" || arg == "--cache") {
            params.use_cache = true;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                return false;
            }
            params.model_path = argv[i];
            params.use_cache = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

static std::vector<ggml_backend_dev_t> get_devices(const rpc_server_params & params) {
    std::vector<ggml_backend_dev_t> devices;
    if (!params.devices.empty()) {
        for (auto device : params.devices) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
            if (dev) {
                devices.push_back(dev);
            } else {
                fprintf(stderr, "error: unknown device: %s\n", device.c_str());
                fprintf(stderr, "available devices:\n");
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    auto * dev = ggml_backend_dev_get(i);
                    size_t free, total;
                    ggml_backend_dev_memory(dev, &free, &total);
                    printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
                }
                return {};
            }
        }
    }

    // Try non-CPU devices first
    if (devices.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                devices.push_back(dev);
            }
        }
    }

    // If there are no accelerators, fallback to CPU device
    if (devices.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev) {
            devices.push_back(dev);
        }
    }

    return devices;
}

static std::string get_exe_directory() {
    namespace fs = std::filesystem;
#ifdef _WIN32
    std::vector<wchar_t> buf(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, buf.data(), (DWORD)buf.size());
    while (len == buf.size()) {
        buf.resize(buf.size() * 2);
        len = GetModuleFileNameW(NULL, buf.data(), (DWORD)buf.size());
    }
    if (len == 0) { return "."; }
    fs::path p(std::wstring(buf.data(), len));
    return p.parent_path().u8string();
#else
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (ec) { return "."; }
    return exe.parent_path().string();
#endif
}

int main(int argc, char * argv[]) {
    std::setlocale(LC_NUMERIC, "C");

    std::string exe_dir = get_exe_directory();
    ggml_backend_load_all_from_path(exe_dir.c_str());

    rpc_server_params params;
    if (!rpc_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    if (params.host != "127.0.0.1") {
        fprintf(stderr, "\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "WARNING: Host ('%s') is != '127.0.0.1'\n", params.host.c_str());
        fprintf(stderr, "         Never expose the RPC server to an open network!\n");
        fprintf(stderr, "         This is an experimental feature and is not secure!\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
    }

    auto devices = get_devices(params);
    if (devices.empty()) {
        fprintf(stderr, "No devices found\n");
        return 1;
    }
    std::string endpoint = params.host + ":" + std::to_string(params.port);
    const char * cache_dir = nullptr;
    std::string cache_dir_str;
    if (params.use_cache) {
        cache_dir_str = fs_get_cache_directory() + "rpc/";
        if (!fs_create_directory_with_parents(cache_dir_str)) {
            fprintf(stderr, "Failed to create cache directory: %s\n", cache_dir_str.c_str());
            return 1;
        }
        cache_dir = cache_dir_str.c_str();
    }

    // Build tensor map from model path
    std::unordered_map<uint64_t, TensorLocation> tensor_map;
    const std::unordered_map<uint64_t, TensorLocation> * tensor_map_ptr = nullptr;
    if (!params.model_path.empty()) {
        if (build_tensor_map(params.model_path, tensor_map)) {
            tensor_map_ptr = &tensor_map;
        } else {
            fprintf(stderr, "Warning: failed to build tensor map from model file\n");
        }
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) {
        fprintf(stderr, "Failed to find RPC backend\n");
        return 1;
    }

    // Note: This function pointer type now includes the tensor_map_ptr parameter
    using start_server_fn_t = void (*)(const char *, const char *, size_t, size_t, ggml_backend_dev_t *, 
                                       const std::unordered_map<uint64_t, TensorLocation> *);
    auto start_server_fn = (start_server_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    if (!start_server_fn) {
        fprintf(stderr, "Failed to obtain RPC backend start server function\n");
        return 1;
    }

    start_server_fn(endpoint.c_str(), cache_dir, params.n_threads, devices.size(), devices.data(), tensor_map_ptr);
    return 0;
}
