#!/usr/bin/env python3
"""Verify vendored harness bytes.

The manifest lists sha256 digests of the public harness files. BOUND_REVISION
is the TensorRelay commit those bytes were copied from. A match proves those
bytes only. It does not record the consumer checkout that later accepts a run,
and it does not claim a fork SHA and a consumer SHA were tested as a pair.
"""
import argparse
import hashlib
import sys
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(text):
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, path = line.split(None, 1)
        entries.append((digest, path))
    if not entries:
        raise ValueError("manifest is empty")
    return entries


def verify(root, claimed_sha, manifest_text, bound_sha):
    """Return a result dict.

    harness_manifest_verified is true only when every digest matches and the
    claimed SHA is the harness source. harness_source_sha is that source.
    The result has no consumer checkout field.
    """
    root = Path(root)
    claimed = (claimed_sha or "").strip()
    bound = (bound_sha or "").strip()
    if len(bound) != 40 or any(ch not in "0123456789abcdef" for ch in bound):
        return {
            "harness_manifest_verified": False,
            "reason": "BOUND_REVISION is not a 40-hex SHA",
        }
    if claimed != bound:
        return {
            "harness_manifest_verified": False,
            "reason": "harness_source_sha does not match BOUND_REVISION",
            "claimed": claimed,
            "bound": bound,
        }
    try:
        entries = load_manifest(manifest_text)
    except ValueError as exc:
        return {"harness_manifest_verified": False, "reason": str(exc)}
    for digest, relative in entries:
        path = root / relative
        if not path.is_file():
            return {
                "harness_manifest_verified": False,
                "reason": f"missing harness file {relative}",
            }
        actual = file_sha256(path)
        if actual != digest:
            return {
                "harness_manifest_verified": False,
                "reason": f"sha256 mismatch for {relative}",
                "expected": digest,
                "actual": actual,
            }
    return {
        "harness_manifest_verified": True,
        "harness_source_sha": bound,
        "reason": "manifest matches vendored harness bytes",
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bound", required=True)
    parser.add_argument("--harness-source-sha", required=True)
    args = parser.parse_args(argv[1:])
    result = verify(
        args.root,
        args.harness_source_sha,
        Path(args.manifest).read_text(),
        Path(args.bound).read_text(),
    )
    verified = bool(result["harness_manifest_verified"])
    if verified:
        print(
            "harness_manifest_verified=true "
            f"harness_source_sha={result['harness_source_sha']} "
            f"reason={result['reason']}"
        )
    else:
        print(f"harness_manifest_verified=false reason={result['reason']}")
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
