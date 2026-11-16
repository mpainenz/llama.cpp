#!/usr/bin/env python3
"""Check load-first discovery from a separate CPU-only executor bundle.

The CPU selector must be found and rejected as a GPU before model loading.
This uses a tiny metadata fixture; no model weights or GPU backend are loaded.
"""
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


class LoadParams(C.Structure):
    # include/tensorrelay_stage_executor.h; preserve its pointer/size layout.
    _fields_ = [
        ('abi_version', C.c_uint32), ('stage_index', C.c_int32),
        ('split_count', C.c_int32), ('slot_count', C.c_uint32),
        ('context_length', C.c_uint32),
        ('shard_path_ptr', C.c_void_p), ('shard_path_len', C.c_size_t),
        ('selected_devices_ptr', C.c_void_p), ('selected_devices_len', C.c_size_t),
        ('spill_layer_count', C.c_uint32),
        ('spill_devices_ptr', C.c_void_p), ('spill_devices_len', C.c_size_t),
        ('cache_type_k_ptr', C.c_void_p), ('cache_type_k_len', C.c_size_t),
        ('cache_type_v_ptr', C.c_void_p), ('cache_type_v_len', C.c_size_t),
    ]


def worker(directory, mode):
    lib = C.CDLL(str(directory / 'libtensorrelay_stage_executor.so'))
    lib.tr_stage_executor_abi_version.restype = C.c_uint32
    assert lib.tr_stage_executor_abi_version() == 11, 'Update the load-parameter layout for the new ABI'
    assert C.sizeof(LoadParams) == 112, 'This check requires the Linux x64 ABI'
    lib.tr_stage_executor_create.argtypes = [C.POINTER(C.c_void_p)]
    lib.tr_stage_executor_destroy.argtypes = [C.c_void_p]
    lib.tr_stage_executor_destroy.restype = None
    lib.tr_stage_executor_load.argtypes = [C.c_void_p, C.POINTER(LoadParams)]
    lib.tr_stage_executor_enumerate_devices.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    lib.tr_stage_executor_last_error.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    lib.tr_stage_executor_last_error.restype = C.c_size_t
    handle = C.c_void_p()
    assert lib.tr_stage_executor_create(C.byref(handle)) == 0
    try:
        if mode == 'enumerate-first':
            size = C.c_size_t()
            assert lib.tr_stage_executor_enumerate_devices(handle, None, 0, C.byref(size)) == 0
        params = LoadParams(abi_version=lib.tr_stage_executor_abi_version(),
                            stage_index=0, split_count=1, slot_count=1, context_length=256)
        buffers = []
        for field, value in [('shard_path', str(directory / 'metadata.gguf')), ('selected_devices', 'CPU')]:
            data = value.encode()
            buffer = C.create_string_buffer(data)
            buffers.append(buffer)
            setattr(params, field + '_ptr', C.addressof(buffer))
            setattr(params, field + '_len', len(data))
        code = lib.tr_stage_executor_load(handle, C.byref(params))
        size = lib.tr_stage_executor_last_error(handle, None, 0)
        error = C.create_string_buffer(size)
        written = lib.tr_stage_executor_last_error(handle, error, size)
        message = error.raw[:written].decode('utf-8', 'replace')
        return {'mode': mode, 'code': code, 'error': message,
                'passed': code == -1 and message.startswith('selected device is not a GPU device: CPU ')}
    finally:
        lib.tr_stage_executor_destroy(handle)


def check(source):
    results = []
    with tempfile.TemporaryDirectory(prefix='tr-backend-discovery-') as temporary:
        root = Path(temporary)
        bundle = root / 'bundle'
        bundle.mkdir()
        empty = root / 'empty'
        empty.mkdir()
        for library in source.iterdir():
            if library.name.startswith(('libtensorrelay_stage_executor.so', 'libllama', 'libggml-base', 'libggml-cpu')) or library.name in ('libggml.so', 'libggml.so.0'):
                if library.is_file():
                    shutil.copy2(library, bundle / library.name)
        def string(value):
            data = value.encode()
            return struct.pack('<Q', len(data)) + data
        header = struct.pack('<IIQQ', 0x46554747, 3, 1, 1)
        header += string('general.architecture') + struct.pack('<I', 8) + string('qwen2')
        header += string('stub') + struct.pack('<IQIQ', 1, 1, 0, 0)
        (bundle / 'metadata.gguf').write_bytes(header + b'\0' * ((-len(header)) % 32) + struct.pack('<f', 0))
        env = dict(os.environ)
        env.pop('GGML_BACKEND_PATH', None)
        env['LD_LIBRARY_PATH'] = str(bundle)
        for mode in ('load-first', 'enumerate-first'):
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(bundle), mode],
                                     cwd=empty, env=env, capture_output=True, text=True, timeout=15, check=True)
            results.append(json.loads(process.stdout.splitlines()[-1]))
    return results


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        print(json.dumps(worker(Path(sys.argv[2]), sys.argv[3])))
    else:
        results = check(Path(sys.argv[1]).resolve())
        print(json.dumps(results, indent=2))
        raise SystemExit(0 if all(result['passed'] for result in results) else 1)
