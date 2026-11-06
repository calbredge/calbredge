"""collector: the runtime version is looked up by the runtime's family.

A precision-qualified label ("tensorrt-fp16") names the same package as its family, so the
legacy result's system_metadata.runtime_version is the real version, not "n/a".
"""

from types import SimpleNamespace

import pytest
from calbredge import collector

pytestmark = pytest.mark.unit


@pytest.fixture
def modules(monkeypatch):
    """importlib.import_module answering for tensorrt and onnxruntime only."""
    versions = {"tensorrt": "10.3.0", "onnxruntime": "1.23.0"}

    def import_module(name):
        if name not in versions:
            raise ImportError(name)
        return SimpleNamespace(__version__=versions[name])

    monkeypatch.setattr(collector.importlib, "import_module", import_module)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("tensorrt-fp16", "10.3.0"),
        ("tensorrt", "10.3.0"),
        ("onnxruntime-cpu", "1.23.0"),
        ("onnxruntime", "1.23.0"),
        ("mock", "n/a"),  # no package to inspect, as before
        ("notaruntime-fp16", "n/a"),
    ],
)
def test_the_version_is_looked_up_by_family(modules, label, expected):
    assert collector._get_runtime_version(label) == expected


def test_collect_metadata_keeps_the_label_and_records_the_real_version(modules):
    metadata = collector.collect_metadata("tensorrt-fp16")
    assert metadata.runtime_name == "tensorrt-fp16"
    assert metadata.runtime_version == "10.3.0"


# --- RKNN: the version of what runs on the device --------------------------------------


@pytest.fixture
def distributions(monkeypatch):
    """importlib.metadata.version answering from a dict the test fills in."""
    installed: dict[str, str] = {}

    def version(name):
        if name not in installed:
            raise collector.importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(collector.importlib.metadata, "version", version)
    return installed


@pytest.mark.parametrize("label", ["rknn", "rknn-fp16", "rknn-int8"])
def test_rknn_version_is_the_rknn_toolkit_lite2_distribution(modules, distributions, label):
    """RKNNAdapter imports rknnlite (rknn-toolkit-lite2). The module named rknn is the x86
    converter, which is not on the device: looking it up there recorded no version at all."""
    distributions["rknn-toolkit-lite2"] = "2.3.2"
    assert collector._get_runtime_version(label) == "2.3.2"


def test_rknn_version_never_comes_from_the_converter(monkeypatch, distributions):
    """rknn-toolkit2 installed without the lite runtime is not a runtime version."""
    distributions["rknn-toolkit2"] = "2.3.2"

    def import_module(name):
        raise AssertionError(f"{name} must not be imported to find the RKNN version")

    monkeypatch.setattr(collector.importlib, "import_module", import_module)
    assert collector._get_runtime_version("rknn-fp16") == "unknown"


def test_rknn_metadata_carries_the_lite_runtime_version(modules, distributions):
    distributions["rknn-toolkit-lite2"] = "2.3.2"
    metadata = collector.collect_metadata("rknn-fp16")
    assert metadata.runtime_name == "rknn-fp16"
    assert metadata.runtime_version == "2.3.2"


def test_other_families_do_not_consult_distribution_metadata(modules, monkeypatch):
    """TensorRT comes from JetPack as a system module: its lookup is the module's own."""

    def version(name):
        raise AssertionError(f"distribution metadata consulted for {name}")

    monkeypatch.setattr(collector.importlib.metadata, "version", version)
    assert collector._get_runtime_version("tensorrt-fp16") == "10.3.0"
    assert collector._get_runtime_version("onnxruntime-cuda") == "1.23.0"
    assert collector._get_runtime_version("mock") == "n/a"
