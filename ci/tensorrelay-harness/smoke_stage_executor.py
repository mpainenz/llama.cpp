#!/usr/bin/env python3
"""Bounded public smoke for libtensorrelay_stage_executor.so.

Writes an observation. It does not compare against a baseline and it does
not classify the device. Layout constants match include/tensorrelay_stage_executor.h
on LP64.
"""
import argparse
import ctypes
import hashlib
import json
import os
import struct
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ABI_VERSION = 11
OK = 0
INPUT_FLAG_TOKEN_IDS = 1 << 2
OUTPUT_FLAG_EOG = 1 << 3
LOAD_SIZE = 112
SLOT_INPUT_SIZE = 40
SLOT_OUTPUT_SIZE = 32
SAMPLER_SIZE = 40
BATCH_SIZE = 64


class LoadParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("stage_index", ctypes.c_int32),
        ("split_count", ctypes.c_int32),
        ("slot_count", ctypes.c_uint32),
        ("context_length", ctypes.c_uint32),
        ("shard_path_ptr", ctypes.c_void_p),
        ("shard_path_len", ctypes.c_size_t),
        ("selected_devices_ptr", ctypes.c_void_p),
        ("selected_devices_len", ctypes.c_size_t),
        ("spill_layer_count", ctypes.c_uint32),
        ("spill_devices_ptr", ctypes.c_void_p),
        ("spill_devices_len", ctypes.c_size_t),
        ("cache_type_k_ptr", ctypes.c_void_p),
        ("cache_type_k_len", ctypes.c_size_t),
        ("cache_type_v_ptr", ctypes.c_void_p),
        ("cache_type_v_len", ctypes.c_size_t),
    ]


class SlotInput(ctypes.Structure):
    _fields_ = [
        ("slot_id", ctypes.c_uint32),
        ("request_epoch", ctypes.c_uint64),
        ("token_offset", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("activation_offset", ctypes.c_uint32),
        ("activation_len", ctypes.c_uint32),
        ("position_start", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class SlotOutput(ctypes.Structure):
    _fields_ = [
        ("slot_id", ctypes.c_uint32),
        ("request_epoch", ctypes.c_uint64),
        ("activation_offset", ctypes.c_uint32),
        ("activation_len", ctypes.c_uint32),
        ("sampled_token_id", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class SamplerParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("request_epoch", ctypes.c_uint64),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("seed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BatchCall(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("input_count", ctypes.c_uint32),
        ("inputs_ptr", ctypes.c_void_p),
        ("input_bytes_ptr", ctypes.c_void_p),
        ("input_bytes_len", ctypes.c_size_t),
        ("output_bytes_ptr", ctypes.c_void_p),
        ("output_bytes_len", ctypes.c_size_t),
        ("outputs_ptr", ctypes.c_void_p),
        ("outputs_len", ctypes.c_uint32),
    ]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path, pin):
    digest = file_sha256(path)
    size = Path(path).stat().st_size
    if digest != pin["sha256"] or size != pin["size"]:
        raise ValueError(f"model digest {digest} size {size} does not match the pin")
    if "/resolve/main/" in pin["url"] or pin["revision"] not in pin["url"]:
        raise ValueError("model URL is not pinned to an immutable revision")
    if Path(path).read_bytes()[:4] != b"GGUF":
        raise ValueError("model is not a GGUF file")
    return digest


def download_model(pin, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(pin["url"], headers={"User-Agent": "llama-stage-executor-smoke"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    temporary.replace(dest)
    return verify_model(dest, pin)


def _bind(library):
    library.tr_stage_executor_abi_version.restype = ctypes.c_uint32
    library.tr_stage_executor_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.tr_stage_executor_create.restype = ctypes.c_int32
    library.tr_stage_executor_destroy.argtypes = [ctypes.c_void_p]
    library.tr_stage_executor_load.argtypes = [ctypes.c_void_p, ctypes.POINTER(LoadParams)]
    library.tr_stage_executor_load.restype = ctypes.c_int32
    library.tr_stage_executor_alloc_slots.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    library.tr_stage_executor_alloc_slots.restype = ctypes.c_int32
    library.tr_stage_executor_capabilities_json.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.tr_stage_executor_capabilities_json.restype = ctypes.c_size_t
    library.tr_stage_executor_enumerate_devices.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.tr_stage_executor_enumerate_devices.restype = ctypes.c_int32
    library.tr_stage_executor_chat_begin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.tr_stage_executor_chat_begin.restype = ctypes.c_int32
    library.tr_stage_executor_configure_slot_sampler.argtypes = [ctypes.c_void_p, ctypes.POINTER(SamplerParams)]
    library.tr_stage_executor_configure_slot_sampler.restype = ctypes.c_int32
    library.tr_stage_executor_execute_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(BatchCall)]
    library.tr_stage_executor_execute_batch.restype = ctypes.c_int32
    library.tr_stage_executor_last_error.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.tr_stage_executor_last_error.restype = ctypes.c_size_t
    library.tr_stage_executor_release_slot.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    library.tr_stage_executor_release_slot.restype = ctypes.c_int32


def _last_error(library, handle):
    needed = library.tr_stage_executor_last_error(handle, None, 0)
    if needed == 0:
        return ""
    buffer = ctypes.create_string_buffer(needed)
    written = library.tr_stage_executor_last_error(handle, buffer, needed)
    return buffer.raw[:written].decode("utf-8", "replace")


def _check(library, handle, code, what):
    if code != OK:
        raise RuntimeError(f"{what} failed code={code} error={_last_error(library, handle)}")


def _execute(library, handle, tokens, position_start, epoch):
    payload = b"".join(struct.pack("<i", token) for token in tokens)
    pinned = ctypes.create_string_buffer(payload)
    slot = SlotInput(
        slot_id=0,
        request_epoch=epoch,
        token_offset=0,
        token_count=len(tokens),
        activation_offset=0,
        activation_len=0,
        position_start=position_start,
        flags=INPUT_FLAG_TOKEN_IDS,
    )
    output_bytes = ctypes.create_string_buffer(4096)
    output = SlotOutput()
    call = BatchCall(
        abi_version=ABI_VERSION,
        input_count=1,
        inputs_ptr=ctypes.addressof(slot),
        input_bytes_ptr=ctypes.addressof(pinned),
        input_bytes_len=len(payload),
        output_bytes_ptr=ctypes.addressof(output_bytes),
        output_bytes_len=len(output_bytes),
        outputs_ptr=ctypes.addressof(output),
        outputs_len=1,
    )
    _check(library, handle, library.tr_stage_executor_execute_batch(handle, ctypes.byref(call)), "execute_batch")
    return output.sampled_token_id, output.flags


def run_smoke(library_path, model_path, settings):
    if ctypes.sizeof(LoadParams) != LOAD_SIZE or ctypes.sizeof(BatchCall) != BATCH_SIZE:
        raise RuntimeError("ABI struct size does not match the LP64 header")
    if ctypes.sizeof(SlotInput) != SLOT_INPUT_SIZE or ctypes.sizeof(SlotOutput) != SLOT_OUTPUT_SIZE:
        raise RuntimeError("ABI slot size does not match the LP64 header")
    if ctypes.sizeof(SamplerParams) != SAMPLER_SIZE:
        raise RuntimeError("ABI sampler size does not match the LP64 header")
    library = ctypes.CDLL(str(library_path))
    _bind(library)
    abi = library.tr_stage_executor_abi_version()
    if abi != ABI_VERSION:
        raise RuntimeError(f"ABI {abi} != {ABI_VERSION}")
    handle = ctypes.c_void_p()
    _check(library, None, library.tr_stage_executor_create(ctypes.byref(handle)), "create")
    try:
        needed = ctypes.c_size_t()
        _check(
            library,
            handle,
            library.tr_stage_executor_enumerate_devices(handle, None, 0, ctypes.byref(needed)),
            "enumerate_devices",
        )
        enumerated = []
        if needed.value:
            raw = ctypes.create_string_buffer(needed.value)
            written = ctypes.c_size_t()
            _check(
                library,
                handle,
                library.tr_stage_executor_enumerate_devices(handle, raw, needed.value, ctypes.byref(written)),
                "enumerate_devices",
            )
            enumerated = json.loads(raw.raw[: written.value].decode("utf-8"))
        selected = ""
        if enumerated and enumerated[0].get("name"):
            selected = str(enumerated[0]["name"])
        shard = Path(model_path).resolve().as_posix().encode()
        selected_bytes = selected.encode()
        cache = b"f16"
        pinned = {
            "shard": ctypes.create_string_buffer(shard),
            "selected": ctypes.create_string_buffer(selected_bytes),
            "spill": ctypes.create_string_buffer(b""),
            "cache_k": ctypes.create_string_buffer(cache),
            "cache_v": ctypes.create_string_buffer(cache),
        }
        params = LoadParams(
            abi_version=ABI_VERSION,
            stage_index=0,
            split_count=1,
            slot_count=1,
            context_length=int(settings["context_length"]),
            shard_path_ptr=ctypes.addressof(pinned["shard"]),
            shard_path_len=len(shard),
            selected_devices_ptr=ctypes.addressof(pinned["selected"]),
            selected_devices_len=len(selected_bytes),
            spill_layer_count=0,
            spill_devices_ptr=ctypes.addressof(pinned["spill"]),
            spill_devices_len=0,
            cache_type_k_ptr=ctypes.addressof(pinned["cache_k"]),
            cache_type_k_len=len(cache),
            cache_type_v_ptr=ctypes.addressof(pinned["cache_v"]),
            cache_type_v_len=len(cache),
        )
        _check(library, handle, library.tr_stage_executor_load(handle, ctypes.byref(params)), "load")
        _check(library, handle, library.tr_stage_executor_alloc_slots(handle, 1), "alloc_slots")
        cap_needed = library.tr_stage_executor_capabilities_json(handle, None, 0)
        cap_buf = ctypes.create_string_buffer(cap_needed)
        cap_written = library.tr_stage_executor_capabilities_json(handle, cap_buf, cap_needed)
        capabilities = json.loads(cap_buf.raw[:cap_written].decode("utf-8"))
        prompt = settings["prompts"][0]
        request = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
        request_buf = ctypes.create_string_buffer(request)
        token_count = ctypes.c_size_t()
        state_len = ctypes.c_size_t()
        _check(
            library,
            handle,
            library.tr_stage_executor_chat_begin(
                handle, request_buf, len(request), None, 0, ctypes.byref(token_count), None, 0, ctypes.byref(state_len)
            ),
            "chat_begin",
        )
        tokens = (ctypes.c_int32 * token_count.value)()
        state = ctypes.create_string_buffer(state_len.value)
        filled_tokens = ctypes.c_size_t()
        filled_state = ctypes.c_size_t()
        _check(
            library,
            handle,
            library.tr_stage_executor_chat_begin(
                handle,
                request_buf,
                len(request),
                tokens,
                token_count.value,
                ctypes.byref(filled_tokens),
                state,
                state_len.value,
                ctypes.byref(filled_state),
            ),
            "chat_begin",
        )
        prompt_tokens = [tokens[index] for index in range(filled_tokens.value)]
        epoch = 1
        sampler = SamplerParams(
            abi_version=ABI_VERSION,
            slot_id=0,
            request_epoch=epoch,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            min_p=float(settings["min_p"]),
            top_k=int(settings["top_k"]),
            seed=int(settings["seed"]),
            reserved=0,
        )
        _check(
            library,
            handle,
            library.tr_stage_executor_configure_slot_sampler(handle, ctypes.byref(sampler)),
            "configure_slot_sampler",
        )
        sampled = []
        n_predict = int(settings["n_predict"])
        token_id, flags = _execute(library, handle, prompt_tokens, 0, epoch)
        sampled.append(int(token_id))
        position = len(prompt_tokens)
        while len(sampled) < n_predict and (flags & OUTPUT_FLAG_EOG) == 0:
            token_id, flags = _execute(library, handle, [token_id], position, epoch)
            sampled.append(int(token_id))
            position += 1
        library.tr_stage_executor_release_slot(handle, 0, epoch)
        return enumerated, selected, capabilities, sampled
    finally:
        library.tr_stage_executor_destroy(handle)


def write_observation(path, settings, pin_sha, fork_sha, harness_source_sha, enumerated, selected, capabilities, sampled):
    """Record the built fork and the vendored harness source. No consumer checkout."""
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runner = os.environ.get("RUNNER_NAME", "unknown")
    body = {
        "fork_sha": fork_sha,
        "harness_source_sha": harness_source_sha,
        "model_id": settings["model_id"],
        "model_sha256": pin_sha,
        "quant": settings["quant"],
        "context_length": settings["context_length"],
        "seed": settings["seed"],
        "temperature": settings["temperature"],
        "top_k": settings["top_k"],
        "top_p": settings["top_p"],
        "min_p": settings["min_p"],
        "n_predict": settings["n_predict"],
        "prompts": settings["prompts"],
        "sampled_token_ids": sampled,
        "enumerated_devices": enumerated,
        "selected_devices": selected,
        "capabilities": capabilities,
        "provenance": {
            "runner": runner,
            "captured_at": captured_at,
            "fork_sha": fork_sha,
            "harness_source_sha": harness_source_sha,
            "model_sha256": pin_sha,
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(f"observation tokens={len(sampled)} runner={runner}")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "run"))
    parser.add_argument("--pin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--settings")
    parser.add_argument("--library")
    parser.add_argument("--fork-sha")
    parser.add_argument("--harness-source-sha")
    parser.add_argument("--out")
    args = parser.parse_args(argv[1:])
    pin = json.loads(Path(args.pin).read_text())
    if args.command == "download":
        digest = download_model(pin, args.model)
        print(f"verified sha256={digest}")
        return 0
    settings = json.loads(Path(args.settings).read_text())
    digest = verify_model(args.model, pin)
    enumerated, selected, capabilities, sampled = run_smoke(args.library, args.model, settings)
    write_observation(
        args.out,
        settings,
        digest,
        args.fork_sha,
        args.harness_source_sha,
        enumerated,
        selected,
        capabilities,
        sampled,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
