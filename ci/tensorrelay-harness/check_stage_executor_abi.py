#!/usr/bin/env python3
"""Check the stage-executor dynamic exports the Rust runtime calls."""
import re
import sys

# Names exported by libtensorrelay_stage_executor.so / the Windows DLL.
# Kept in sync with client/src-tauri references to tr_stage_executor_*.
REQUIRED = (
    "tr_stage_executor_abi_version",
    "tr_stage_executor_activation_descriptor_json",
    "tr_stage_executor_alloc_slots",
    "tr_stage_executor_benchmark",
    "tr_stage_executor_capabilities_json",
    "tr_stage_executor_chat_begin",
    "tr_stage_executor_chat_parse",
    "tr_stage_executor_configure_slot_sampler",
    "tr_stage_executor_create",
    "tr_stage_executor_destroy",
    "tr_stage_executor_enumerate_devices",
    "tr_stage_executor_execute_batch",
    "tr_stage_executor_last_error",
    "tr_stage_executor_load",
    "tr_stage_executor_read_diagnostics",
    "tr_stage_executor_release_slot",
)


def missing_symbols(nm_or_dumpbin_text):
    """Return required symbols absent from nm -D or dumpbin /exports text."""
    present = set(re.findall(r"\btr_stage_executor_[A-Za-z0-9_]+", nm_or_dumpbin_text))
    return [name for name in REQUIRED if name not in present]


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: check_stage_executor_abi.py SYMBOL_TEXT")
    text = open(argv[1], encoding="utf-8", errors="replace").read()
    missing = missing_symbols(text)
    if missing:
        print("missing stage-executor exports:", ", ".join(missing), file=sys.stderr)
        raise SystemExit(1)
    print(f"stage-executor ABI exports verified ({len(REQUIRED)})")


if __name__ == "__main__":
    main(sys.argv)
