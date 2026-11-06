"""Write the v2 run bundle: <bundle_dir>/<result_id>/ (docs/RESULT_SCHEMA.md §6).

    result.json          the v2 result (or result.invalid.json + errors)
    samples.jsonl        one {"iteration", "latency_ms"} object per measured iteration
    environment.json     versions, command line (redacted), allowlisted env vars
    logs/calbredge.log   the progress lines this run printed

The bundle is written into a hidden temporary directory and renamed into place,
so a crash never leaves a partial bundle under the result_id. Nothing here
writes to stdout.

Secrets never reach the bundle: environment.json holds only allowlisted
environment variables and a redacted command line, and every file is scrubbed
of known secret values (the resolved API key, and the value of any
environment variable whose name looks like a credential) before it is written.
"""

import json
import os
import platform
import re
import sys
from collections.abc import Iterable
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

DEFAULT_BUNDLE_DIR = Path.home() / ".calbredge" / "runs"
REDACTED = "<redacted>"

# Only these environment variables (or ones with these prefixes) are recorded,
# and never one whose name also looks like a credential.
ENV_ALLOWLIST = frozenset(
    {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "LD_LIBRARY_PATH", "PYTHONHASHSEED"}
)
ENV_ALLOWLIST_PREFIXES = ("ORT_", "TRT_", "TENSORRT_")

# Flag and env-var names that carry credentials. Deliberately broad for names
# (over-redacting a flag costs nothing).
_SECRET_NAME = re.compile(r"key|token|secret|passw|credential|auth", re.IGNORECASE)
# Env vars whose *values* are scrubbed from every bundle file. Narrower: a
# scrubbed value is replaced wherever it appears, so harmless names like
# SSH_AUTH_SOCK or XDG_SESSION_CLASS=user must not match.
_SECRET_ENV_VALUE_NAME = re.compile(
    r"api[_-]?key|access[_-]?key|private[_-]?key|secret|token|passw|credential", re.IGNORECASE
)
_MIN_SCRUB_LEN = 8
# name=value pairs inside a single argument (e.g. a URL query string).
_SECRET_PAIR = re.compile(
    r"((?:[A-Za-z0-9_-]*(?:key|token|secret|passw|credential|auth)[A-Za-z0-9_-]*)=)[^&\s]+",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"(://)[^/@\s]+@")


def secret_values(extra: Iterable[str | None] = ()) -> list[str]:
    """Values to scrub: `extra` (e.g. the resolved API key) plus credential-named env vars."""
    values = [v for v in extra if v]
    values += [v for k, v in os.environ.items() if _SECRET_ENV_VALUE_NAME.search(k) and v]
    # Longest first, so a secret containing another is replaced whole. The
    # explicit extras (the API key) are always scrubbed regardless of length.
    explicit = {v for v in extra if v}
    return sorted(
        {v for v in values if v in explicit or len(v) >= _MIN_SCRUB_LEN}, key=len, reverse=True
    )


def redact_argv(argv: list[str]) -> list[str]:
    """Redact values of credential-named flags, name=value pairs and URL userinfo."""
    out: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            out.append(REDACTED)
            redact_next = False
            continue
        if arg.startswith("-"):
            name, eq, _ = arg.partition("=")
            if _SECRET_NAME.search(name):
                if eq:
                    out.append(f"{name}={REDACTED}")
                else:
                    out.append(arg)
                    redact_next = True
                continue
        arg = _SECRET_PAIR.sub(rf"\1{REDACTED}", arg)
        out.append(_URL_USERINFO.sub(rf"\1{REDACTED}@", arg))
    return out


def _allowlisted_env() -> dict[str, str]:
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if (k in ENV_ALLOWLIST or k.startswith(ENV_ALLOWLIST_PREFIXES))
        and not _SECRET_NAME.search(k)
    }


def build_environment(argv: list[str], calbredge_version: str) -> dict[str, Any]:
    packages = {
        dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]
    }
    uname = platform.uname()
    return {
        "calbredge_version": calbredge_version,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "command_line": redact_argv(argv),
        "environment_variables": _allowlisted_env(),
        "packages": dict(sorted(packages.items(), key=lambda kv: kv[0].lower())),
    }


def _scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    return text


def result_json_text(result: dict[str, Any]) -> str:
    """result.json exactly as the bundle writes it (a decision's candidate_result_digest)."""
    return json.dumps(result, indent=2) + "\n"


def write_bundle(
    bundle_dir: Path,
    *,
    result_id: str,
    result: dict[str, Any],
    errors: list[str],
    samples_ms: tuple[float, ...],
    environment: dict[str, Any],
    log_lines: list[str],
    secrets: list[str],
) -> Path:
    """Write the bundle atomically and return its final path."""
    final = bundle_dir / result_id
    if final.exists():
        raise FileExistsError(f"run bundle already exists: {final}")
    tmp = bundle_dir / f".{result_id}.partial"
    (tmp / "logs").mkdir(parents=True, exist_ok=False)

    def write(rel: str, text: str) -> None:
        (tmp / rel).write_text(_scrub(text, secrets), encoding="utf-8")

    if errors:
        write(
            "result.invalid.json", json.dumps({"errors": errors, "result": result}, indent=2) + "\n"
        )
    else:
        write("result.json", result_json_text(result))
    # json.dumps writes floats with repr(): exact round-trip, so statistics
    # recomputed from this file equal the recorded ones.
    write(
        "samples.jsonl",
        "".join(
            json.dumps({"iteration": i, "latency_ms": s}) + "\n" for i, s in enumerate(samples_ms)
        ),
    )
    write("environment.json", json.dumps(environment, indent=2) + "\n")
    write("logs/calbredge.log", "".join(line + "\n" for line in log_lines))
    os.replace(tmp, final)
    return final
