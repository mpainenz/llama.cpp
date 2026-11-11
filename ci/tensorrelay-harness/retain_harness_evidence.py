#!/usr/bin/env python3
"""Retain sanitized harness logs and explicit step outcomes.

Identity verification is a separate step. This module does not decide that
the vendored bytes matched. It records configure, build, export, readelf, and
smoke outcomes, and it writes harness_run_succeeded only when every one of
those steps succeeded.
"""
import argparse
import json
import re
import sys
from pathlib import Path

STEPS = ("configure", "build", "export", "readelf", "smoke")
ALLOWED = {"success", "failure", "skipped", "cancelled"}

_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\b(authorization:\s*bearer\s+)\S+"),
    re.compile(r"(?i)\b((?:api[_-]?key|token|password|secret)(?:\s*[=:]\s*))\S+"),
    re.compile(r"::add-mask::\S*"),
)


def sanitize(text):
    """Redact credential-shaped strings. Build flags stay intact."""
    if text is None:
        return ""

    def replace(match):
        if match.lastindex:
            return match.group(1) + "[REDACTED]"
        return "[REDACTED]"

    redacted = text
    for pattern in _PATTERNS:
        redacted = pattern.sub(replace, redacted)
    return redacted


def outcomes(values):
    """Map step outcomes. An unknown value is not success."""
    recorded = {}
    for name in STEPS:
        value = (values.get(name) or "").strip()
        recorded[name] = value if value in ALLOWED else "skipped"
    recorded["harness_run_succeeded"] = all(recorded[name] == "success" for name in STEPS)
    return recorded


def summary(body):
    """Markdown that repeats the recorded outcomes and does not upgrade a failure."""
    lines = ["## Build and smoke outcomes", ""]
    for name in STEPS:
        lines.append(f"- {name}: {body[name]}")
    succeeded = body["harness_run_succeeded"] is True
    lines.append(f"- harness_run_succeeded: {'true' if succeeded else 'false'}")
    if not succeeded:
        lines.append("- result: a measured step did not succeed")
    return "\n".join(lines) + "\n"


def retain(dest, logs, measurements, outcome_values):
    """Copy sanitized logs and raw measurements. Missing files stay missing."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for name, source in list(logs) + list(measurements):
        path = Path(source) if source else None
        if path is None or not path.is_file():
            missing.append(name)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        (dest / name).write_text(sanitize(text))
        copied.append(name)
    body = outcomes(outcome_values)
    body["copied"] = copied
    body["missing"] = missing
    (dest / "outcomes.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--configure-log")
    parser.add_argument("--build-log")
    parser.add_argument("--export-log")
    parser.add_argument("--readelf-dyn-syms")
    parser.add_argument("--readelf-needed")
    parser.add_argument("--readelf-version")
    parser.add_argument("--configure-outcome", default="")
    parser.add_argument("--build-outcome", default="")
    parser.add_argument("--export-outcome", default="")
    parser.add_argument("--readelf-outcome", default="")
    parser.add_argument("--smoke-outcome", default="")
    parser.add_argument("--summary")
    args = parser.parse_args(argv[1:])
    body = retain(
        args.dest,
        (
            ("cmake-configure.log", args.configure_log),
            ("cmake-build.log", args.build_log),
            ("executor-exports.txt", args.export_log),
        ),
        (
            ("readelf-dyn-syms.txt", args.readelf_dyn_syms),
            ("readelf-needed.txt", args.readelf_needed),
            ("readelf-version-info.txt", args.readelf_version),
        ),
        {
            "configure": args.configure_outcome,
            "build": args.build_outcome,
            "export": args.export_outcome,
            "readelf": args.readelf_outcome,
            "smoke": args.smoke_outcome,
        },
    )
    text = summary(body)
    if args.summary:
        Path(args.summary).write_text(text)
    sys.stdout.write(text)
    print(f"harness_run_succeeded={'true' if body['harness_run_succeeded'] else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
