"""The quickstart: the model fetcher's hash check, and the docs' commands.

The model fetcher (calbredge/example_model.py, `python3 -m calbredge.example_model`) is
tested with urlopen stubbed (no network).
Every `calbredge` command in docs/QUICKSTART.md and README.md must use only
flags its subcommand's --help lists, so the docs can't drift from the CLI.
"""

import hashlib
import re
import shlex
import urllib.error
from pathlib import Path

import calbredge.example_model as fetcher
import pytest
from calbredge.cli import main
from click.testing import CliRunner

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "docs" / "QUICKSTART.md", ROOT / "README.md"]


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Serve bytes at https URLs from local files; any other https URL is a URLError."""
    files: dict[str, Path] = {}

    def urlopen(url, timeout):
        if url not in files:
            raise urllib.error.URLError(f"no such URL: {url}")
        return files[url].open("rb")

    def add(content: bytes = b"model bytes", name: str = "source.onnx") -> tuple[str, str]:
        src = tmp_path / name
        src.write_bytes(content)
        url = f"https://models.example/{name}"
        files[url] = src
        return url, hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", urlopen)
    return add


def test_a_matching_download_is_kept(tmp_path, serve):
    url, sha = serve()
    dest = tmp_path / "models" / "m.onnx"
    assert fetcher.fetch(url, dest, sha) == "downloaded"
    assert dest.read_bytes() == b"model bytes"
    assert not list(dest.parent.glob("*.partial"))
    assert fetcher.fetch(url, dest, sha) == "present"


def test_a_mismatch_fails_and_deletes_the_file(tmp_path, serve):
    url, _ = serve()
    dest = tmp_path / "models" / "m.onnx"
    with pytest.raises(fetcher.HashMismatchError, match="expected " + "0" * 64):
        fetcher.fetch(url, dest, "0" * 64, attempts=1)
    assert not dest.exists()
    assert not list(dest.parent.glob("*"))  # the partial download is gone too


def test_wrong_bytes_already_at_the_destination_are_replaced(tmp_path, serve):
    url, sha = serve()
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"truncated")
    assert fetcher.fetch(url, dest, sha) == "downloaded"
    assert dest.read_bytes() == b"model bytes"


def test_a_failed_download_leaves_nothing(tmp_path, serve):
    dest = tmp_path / "models" / "m.onnx"
    with pytest.raises(urllib.error.URLError):
        fetcher.fetch("https://models.example/missing.onnx", dest, "0" * 64, attempts=1)
    assert not dest.exists()
    assert not list(dest.parent.glob("*"))


@pytest.mark.parametrize("scheme", ["file", "http"])
def test_a_url_that_isnt_https_is_refused_and_nothing_is_written(
    tmp_path, monkeypatch, capsys, scheme
):
    src = tmp_path / "source.onnx"
    src.write_bytes(b"model bytes")
    url = src.as_uri() if scheme == "file" else "http://models.example/source.onnx"
    sha = hashlib.sha256(b"model bytes").hexdigest()

    def urlopen(url, timeout):
        raise AssertionError(f"opened {url}")

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", urlopen)
    dest = tmp_path / "models" / "m.onnx"
    with pytest.raises(fetcher.InsecureURLError, match="only https"):
        fetcher.fetch(url, dest, sha)
    assert not dest.parent.exists()

    monkeypatch.setattr(fetcher, "MODEL_URL", url)
    monkeypatch.setattr(fetcher, "MODEL_SHA256", sha)
    assert fetcher.main(["--dest", str(dest)]) == 1
    assert f"refusing to download {url}: only https URLs are allowed" in capsys.readouterr().err
    assert not dest.parent.exists()


def test_main_exits_1_on_a_mismatch(tmp_path, monkeypatch, capsys, serve):
    url, _ = serve()
    monkeypatch.setattr(fetcher, "MODEL_URL", url)
    monkeypatch.setattr(fetcher, "MODEL_SHA256", "f" * 64)
    monkeypatch.setattr(fetcher, "fetch", _one_attempt(fetcher.fetch))
    assert fetcher.main(["--dest", str(tmp_path / "m.onnx")]) == 1
    assert "not fetched" in capsys.readouterr().err
    assert not (tmp_path / "m.onnx").exists()


def _one_attempt(fetch):
    return lambda url, dest, sha: fetch(url, dest, sha, attempts=1)


def test_the_model_is_pinned_by_commit_and_hash():
    assert fetcher.MODEL_URL.startswith("https://")
    assert re.search(r"/raw/[0-9a-f]{40}/", fetcher.MODEL_URL)
    assert re.fullmatch(r"[0-9a-f]{64}", fetcher.MODEL_SHA256)
    quickstart = (ROOT / "docs" / "QUICKSTART.md").read_text()
    assert fetcher.MODEL_SHA256[:8] in quickstart
    assert "Apache 2.0" in quickstart


def _commands(path: Path) -> list[list[str]]:
    """Every `calbredge ...` command in the doc's shell code blocks, continuation lines joined.

    Shell blocks are untagged or `bash`; other blocks (`text`, example output) are skipped.
    """
    commands = []
    blocks = re.findall(r"^[ \t]*```(\w*)\n(.*?)^[ \t]*```", path.read_text(), re.DOTALL | re.M)
    for lang, block in blocks:
        if lang not in ("", "bash"):
            continue
        for line in block.replace("\\\n", " ").splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("calbredge "):
                commands.append(shlex.split(line)[1:])
    return commands


def _split(args: list[str]) -> tuple[list[str], list[str]]:
    """(subcommand path, flags) for one command line."""
    path = []
    for arg in args:
        if arg.startswith("-"):
            break
        path.append(arg)
    flags = [a.split("=", 1)[0] for a in args if a.startswith("--")]
    return path, flags


def test_quickstart_covers_the_whole_path():
    """One walkthrough per target, each from setup to its first run. A Jetson and an RK3588:
    install from a clone with the constraints file, set up the clock unit, check the target,
    init, then run, the RK3588 after converting on an x86 host. On x86, where init can't run,
    check the host and run with a bundled protocol."""
    text = DOCS[0].read_text()
    assert 'pip install -c constraints/jetson-jp6.2-py310.txt ./protocol "./agent[jetson]"' in text
    assert 'pip install -c constraints/rk3588-jammy-py310.txt ./protocol "./agent[rk3588]"' in text
    for unit in ("calbredge-jetson-clocks.service", "calbredge-rk3588-clocks.service"):
        assert f"sudo cp examples/systemd/{unit} /etc/systemd/system/" in text
        assert (ROOT / "examples" / "systemd" / unit).is_file()
    subcommands = [" ".join(_split(args)[0]) for args in _commands(DOCS[0])]
    assert 'pip install ./protocol "./agent[onnxruntime]"' in text
    assert subcommands == [
        *["doctor", "init", "init", "run"],  # NVIDIA Jetson
        *["convert", "doctor", "init", "run"],  # Rockchip RK3588, RKNN
        *["doctor", "run"],  # x86_64 Linux, ONNX Runtime
    ]
    targets = re.findall(r"^## (.+)$", text, re.M)
    assert targets[2:5] == ["NVIDIA Jetson", "Rockchip RK3588, RKNN", "x86_64 Linux, ONNX Runtime"]
    assert "[TO COME]" not in text
    bundled = re.findall(r"--protocol (protocol/\S+\.yaml)", text)
    assert bundled
    assert all((ROOT / p).is_file() for p in bundled)


def check_documented_flags(doc: Path) -> None:
    """Every command in the doc exists, and so does every flag it is shown with."""
    commands = _commands(doc)
    assert commands, f"no calbredge commands found in {doc.name}"
    for args in commands:
        path, flags = _split(args)
        result = CliRunner().invoke(main, [*path, "--help"])
        assert result.exit_code == 0, f"`calbredge {' '.join(path)}` isn't a command"
        for flag in flags:
            assert flag in result.output, f"`calbredge {' '.join(path)}` has no {flag}"


@pytest.mark.parametrize("doc", DOCS, ids=[d.name for d in DOCS])
def test_every_documented_flag_exists(doc):
    check_documented_flags(doc)


def test_the_quickstart_names_the_fields_two_runs_are_compared_on():
    """The fields the compare section lists are fields a v2 result has."""
    from calbredge_protocol.result_v2 import Candidate, ProtocolInfo, Target

    groups = {"protocol": ProtocolInfo, "candidate": Candidate, "target": Target}
    section = DOCS[0].read_text().split("**Comparing two runs.**", 1)[1].split("\n## ", 1)[0]
    names = re.findall(r"`((?:protocol|candidate|target)\.\w+)`", section)
    assert len(names) >= 8
    for name in names:
        group, field = name.split(".")
        assert field in groups[group].model_fields, name


@pytest.mark.parametrize("path", [*DOCS, ROOT / "docs" / "JETSON_SETUP.md"], ids=lambda p: p.name)
def test_no_public_pypi_install(path):
    text = path.read_text()
    assert "pip install calbredge" not in text
