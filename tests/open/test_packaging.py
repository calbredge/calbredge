"""Package metadata of the open pair: one version, the agent's exact pin on its protocol, no
dependency on the gate, and the Apache licence and notice in each wheel and sdist.

These read the installed distributions' metadata, so an editable install that
predates a pyproject change fails here until it is reinstalled.
"""

import hashlib
import importlib.metadata as md
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import calbredge
import pytest
from calbredge.cli import main
from click.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]
# https://www.apache.org/licenses/LICENSE-2.0.txt, byte for byte.
APACHE_2_0_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
# agent/NOTICE and protocol/NOTICE, byte for byte: the product name and the copyright line.
NOTICE_SHA256 = "8fdffae34336d463b3f90ca772d6ba4c7f1ef6b16d75d32c727c8969a5345530"
OPEN_PACKAGES = [("agent", "calbredge"), ("protocol", "calbredge_protocol")]

pytestmark = pytest.mark.unit


def test_version_comes_from_package_metadata():
    assert calbredge.__version__ == md.version("calbredge")


def test_cli_version_reports_the_installed_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"calbredge, version {md.version('calbredge')}"


def test_agent_pins_exactly_the_protocol_it_ships_with():
    assert md.version("calbredge") == md.version("calbredge-protocol")
    pin = f"calbredge-protocol=={md.version('calbredge-protocol')}"
    assert pin in (md.requires("calbredge") or [])


@pytest.mark.parametrize("dist", ["calbredge", "calbredge-protocol"])
def test_python_floor_is_310(dist):
    assert md.metadata(dist)["Requires-Python"] == ">=3.10"


def test_the_open_packages_never_require_the_gate():
    for dist in ("calbredge", "calbredge-protocol"):
        assert not [r for r in md.requires(dist) or [] if r.startswith("calbredge-gate")]


def test_jetson_extra_installs_pycuda():
    assert 'pycuda; extra == "jetson"' in (md.requires("calbredge") or [])


def _copy_source(name: str, tmp_path: Path) -> Path:
    """A clean copy of one package's source, so a build never writes into the checkout."""
    src = tmp_path / name
    shutil.copytree(
        REPO / name,
        src,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )
    return src


def _build_wheel(src: Path, out: Path) -> Path:
    build = "import sys; from setuptools import build_meta as b; print(b.build_wheel(sys.argv[1]))"
    done = subprocess.run(
        [sys.executable, "-c", build, str(out)], cwd=src, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return out / done.stdout.strip().splitlines()[-1]


def test_a_stale_build_directory_never_reaches_the_agent_wheel(tmp_path):
    """Only calbredge/ and its dist-info, even with a stray build/lib/calbredge/ in agent/.

    Unrestricted package discovery once packaged build/lib/calbredge/*.py from a
    stale build as a second copy of the package (found on a Jetson).
    """
    src = _copy_source("agent", tmp_path)
    stray = src / "build" / "lib" / "calbredge"
    stray.mkdir(parents=True)
    for name in ("__init__.py", "cli.py"):
        shutil.copy(src / "calbredge" / name, stray / name)
    wheel = _build_wheel(src, tmp_path / "dist")
    dist_info = f"calbredge-{md.version('calbredge')}.dist-info/"
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    assert "calbredge/cli.py" in names
    assert [n for n in names if not n.startswith(("calbredge/", dist_info))] == []


def _build_sdist(src: Path, out: Path) -> Path:
    build = "import sys; from setuptools import build_meta as b; print(b.build_sdist(sys.argv[1]))"
    done = subprocess.run(
        [sys.executable, "-c", build, str(out)], cwd=src, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return out / done.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(("source", "dist"), OPEN_PACKAGES)
def test_the_licence_is_the_apache_text_unmodified(source, dist):
    assert hashlib.sha256((REPO / source / "LICENSE").read_bytes()).hexdigest() == APACHE_2_0_SHA256
    assert hashlib.sha256((REPO / source / "NOTICE").read_bytes()).hexdigest() == NOTICE_SHA256


@pytest.mark.parametrize(("source", "dist"), OPEN_PACKAGES)
def test_the_wheel_carries_the_apache_licence_and_the_notice(source, dist, tmp_path):
    """LICENSE and NOTICE in dist-info, licence metadata, and never-upload."""
    wheel = _build_wheel(_copy_source(source, tmp_path), tmp_path / "dist")
    dist_info = f"{dist}-{md.version('calbredge')}.dist-info"
    with zipfile.ZipFile(wheel) as zf:
        licence_files = sorted(n for n in zf.namelist() if n.startswith(f"{dist_info}/licenses/"))
        licence = zf.read(f"{dist_info}/licenses/LICENSE")
        notice = zf.read(f"{dist_info}/licenses/NOTICE")
        metadata = zf.read(f"{dist_info}/METADATA").decode()
    assert licence_files == [f"{dist_info}/licenses/LICENSE", f"{dist_info}/licenses/NOTICE"]
    assert hashlib.sha256(licence).hexdigest() == APACHE_2_0_SHA256
    assert hashlib.sha256(notice).hexdigest() == NOTICE_SHA256
    headers = metadata.split("\n\n", 1)[0].splitlines()
    assert "License-Expression: Apache-2.0" in headers
    assert [h for h in headers if h.startswith("License-File:")] == [
        "License-File: LICENSE",
        "License-File: NOTICE",
    ]
    assert "Classifier: Private :: Do Not Upload" in headers
    assert not [h for h in headers if h.startswith("Classifier: License ::")]


@pytest.mark.parametrize(("source", "dist"), OPEN_PACKAGES)
def test_the_sdist_carries_the_apache_licence_and_the_notice(source, dist, tmp_path):
    sdist = _build_sdist(_copy_source(source, tmp_path), tmp_path / "dist")
    root = f"{dist}-{md.version('calbredge')}"
    with tarfile.open(sdist) as tf:
        licence = tf.extractfile(f"{root}/LICENSE")
        notice = tf.extractfile(f"{root}/NOTICE")
        info = tf.extractfile(f"{root}/PKG-INFO")
        assert licence is not None
        assert notice is not None
        assert info is not None
        assert hashlib.sha256(licence.read()).hexdigest() == APACHE_2_0_SHA256
        assert hashlib.sha256(notice.read()).hexdigest() == NOTICE_SHA256
        headers = info.read().decode().split("\n\n", 1)[0].splitlines()
    assert "License-Expression: Apache-2.0" in headers
    assert "Classifier: Private :: Do Not Upload" in headers
