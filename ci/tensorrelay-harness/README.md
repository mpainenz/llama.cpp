# Stage-executor smoke

This directory is the public measurement harness for branch `tensorrelay`.

`smoke_stage_executor.py` loads `libtensorrelay_stage_executor.so`, runs one fixed prompt, and writes `observation.json` (device list, token ids, settings, fork SHA, harness source SHA, model digest, runner, timestamp). It does not decide whether a run passes and it does not record a consumer checkout. TensorRelay compares that observation in private.

`verify_manifest.py` checks the harness files against `MANIFEST.sha256` and `BOUND_REVISION`. A match sets `harness_manifest_verified` and `harness_source_sha`. Those fields prove the vendored bytes only. `workflow_dispatch` and `push` are both already admitted for this repository on `refs/heads/tensorrelay` (admission commit `8f24d0226a232d3cbf8b74e1d2125c9fe4921d3b`).

Licence: MIT, the same licence as this repository. Scope: the files added by the TensorRelay format-patch and nothing else.
