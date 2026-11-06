"""Download the example model and verify it against a pinned SHA-256.

Open code, used by `calbredge init --example`, and runnable on its own:

    python3 -m calbredge.example_model [--dest models/mobilenetv2-12.onnx]

The model is MobileNetV2 1.0 (ImageNet classifier, 224x224, opset 12) from the
ONNX Model Zoo (github.com/onnx/models), licensed Apache 2.0. It is downloaded,
never bundled: Calbredge doesn't redistribute it. The URL is pinned to one
commit of that repository and the file to one SHA-256, so every quickstart
measures exactly the same bytes. A download that doesn't match is deleted.

Standard library only.
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://github.com/onnx/models/raw/4f43949841cb55a0b98dc8fcd045431ccafd9f96/"
    "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
)
MODEL_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
DEFAULT_DEST = Path("models") / "mobilenetv2-12.onnx"
ATTEMPTS = 3


class HashMismatchError(RuntimeError):
    """The downloaded file isn't the pinned model."""


class InsecureURLError(ValueError):
    """The URL isn't https, so it is never opened."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, sha256: str, *, attempts: int = ATTEMPTS) -> str:
    """Download ``url`` to ``dest`` and verify it; return "present" or "downloaded".

    An existing file with the right hash is kept. A download is written to a
    ``.partial`` file and only renamed into place once its hash matches;
    otherwise it is deleted and, after the last attempt, HashMismatchError
    is raised (a network error after the last attempt is re-raised). A URL
    that isn't https raises InsecureURLError before anything is touched.
    """
    if urllib.parse.urlsplit(url).scheme != "https":
        raise InsecureURLError(f"refusing to download {url}: only https URLs are allowed")
    if dest.is_file():
        if sha256_of(dest) == sha256:
            return "present"
        dest.unlink()  # wrong bytes where the model should be: replace them
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:  # noqa: S310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected -- https-only, enforced above; commit-pinned URL; SHA-256 verified
                while chunk := response.read(1 << 20):
                    f.write(chunk)
            actual = sha256_of(partial)
            if actual == sha256:
                os.replace(partial, dest)
                return "downloaded"
            partial.unlink()
            if attempt == attempts:
                raise HashMismatchError(
                    f"downloaded file has SHA-256 {actual}, expected {sha256}; deleted it"
                )
        except OSError:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
        time.sleep(attempt)
    raise AssertionError("unreachable")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download the quickstart model and verify its pinned SHA-256."
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    try:
        how = fetch(MODEL_URL, args.dest, MODEL_SHA256)
    except (HashMismatchError, InsecureURLError, OSError) as exc:
        print(f"quickstart model not fetched: {exc}", file=sys.stderr)
        return 1
    print(f"{args.dest}: {how}, SHA-256 {MODEL_SHA256} verified")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
