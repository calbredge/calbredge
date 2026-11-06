"""Canonical forms and digests for the v2 result schema (docs/RESULT_SCHEMA.md §2).

Everything that is compared or hashed in v2 goes through this module, so
there is exactly one definition of each canonical form:

- ``canonical_identifier``: lowercase, runs of non-alphanumerics collapsed to
  ``_``. Same rule the Jetson nvpmodel reader already records power_mode with
  (protocol/calbredge_protocol/adapters.py ``_get_nvpmodel_mode``), so ``MAXN SUPER`` and
  ``maxn_super`` are one value.
- ``canonical_json``: RFC 8785 (JSON Canonicalization Scheme). Baseline
  declarations are digested with it and committed to customer repositories,
  so its output for a given input must never change across releases;
  tests/open/test_engine_cache.py's test_build_key_is_stable pins a digest of it.
"""

import hashlib
import math
import re
from decimal import Decimal
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_LABEL = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Largest integer an IEEE-754 double represents exactly. RFC 8785 serializes
# every number as a double, so an int beyond this would silently change value.
_MAX_EXACT_INT = 2**53


def canonical_identifier(raw: str) -> str:
    """Return the canonical form of a free-text identifier.

    Lowercases, drops NUL bytes (``/proc/device-tree/model`` is
    NUL-terminated), collapses every run of non-alphanumeric characters to a
    single ``_`` and strips leading/trailing ``_``.
    """
    return _NON_ALNUM.sub("_", raw.replace("\x00", "").lower()).strip("_")


def is_canonical_identifier(value: str) -> bool:
    return bool(value) and canonical_identifier(value) == value


def is_digest(value: str) -> bool:
    return bool(_DIGEST.match(value))


def is_git_commit(value: str) -> bool:
    return bool(_GIT_COMMIT.match(value))


def is_runtime_label(value: str) -> bool:
    return bool(_RUNTIME_LABEL.match(value))


def runtime_family(runtime_label: str) -> str:
    """``tensorrt-fp16`` -> ``tensorrt``; ``onnxruntime`` -> ``onnxruntime``."""
    return runtime_label.split("-", 1)[0]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize ``value`` per RFC 8785 and return UTF-8 bytes.

    Accepts only JSON data: None, bool, int, float, str, list/tuple, and dicts
    with str keys. NaN and ±Infinity are rejected (they have no JSON form).
    """
    return _serialize(value).encode("utf-8")


def _serialize(value: Any) -> str:
    # bool before int: bool is a subclass of int.
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > _MAX_EXACT_INT:
            raise ValueError(f"integer {value} is not exactly representable as an IEEE-754 double")
        return str(value)
    if isinstance(value, float):
        return _serialize_number(value)
    if isinstance(value, str):
        return _serialize_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize(v) for v in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"object keys must be strings, got {type(key).__name__}")
        # RFC 8785 §3.2.3: sort by UTF-16 code units, not code points.
        keys = sorted(value, key=lambda k: k.encode("utf-16-be"))
        return "{" + ",".join(_serialize_string(k) + ":" + _serialize(value[k]) for k in keys) + "}"
    raise TypeError(f"not JSON data: {type(value).__name__}")


def _serialize_number(x: float) -> str:
    """ECMAScript Number.prototype.toString, as RFC 8785 §3.2.2.3 requires."""
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"{x!r} has no JSON representation")
    if x == 0:
        return "0"  # covers -0.0
    sign = "-" if x < 0 else ""
    # repr() is the shortest round-tripping decimal, the same digit string
    # ECMAScript picks; only the layout rules below differ from Python's.
    dec = Decimal(repr(abs(x))).normalize()
    _, digit_tuple, exponent = dec.as_tuple()
    if not isinstance(exponent, int):  # only NaN/Infinity have a str exponent; rejected above
        raise ValueError(f"{x!r} has no JSON representation")
    digits = "".join(str(d) for d in digit_tuple)
    k = len(digits)
    n = exponent + k  # value = 0.<digits> * 10**n
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
        body = f"{mantissa}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + body


_SHORT_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _serialize_string(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch in _SHORT_ESCAPES:
            out.append(_SHORT_ESCAPES[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)
