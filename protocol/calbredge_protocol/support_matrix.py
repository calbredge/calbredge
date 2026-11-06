"""The v2 support matrix: which runtime/target combinations may produce baselines.

docs/SUPPORTED_PLATFORMS.md is the human-readable form of this table;
the two change together. Keyed by (runtime_family [+ execution provider],
os, host device class, accelerator, precision, workload_type), so a new
target (Hailo, a Snapdragon board, Windows-on-ARM via ONNX Runtime's QNN
execution provider) is a new row, not a redesign.

Rows are matched in order and the first match wins, so a specific row
(a precision, a workload type) sits above the general one it narrows.
``*`` matches anything, including an unknown (None) value.

Status:
- ``supported``: results can become v2 baselines and be gated.
- ``results_only``: v2 results are recorded, but ``baseline create`` refuses
  them until the listed gap is closed and validated on hardware.
- ``unsupported``: the combination cannot produce trustworthy v2 results yet.

A combination with no row is treated as unsupported.
"""

from dataclasses import dataclass
from typing import Literal

Status = Literal["supported", "results_only", "unsupported"]
ANY = "*"


@dataclass(frozen=True)
class MatrixRow:
    runtime_family: str
    execution_provider: str | None  # ONNX Runtime only (e.g. "cpu", "cuda"); None elsewhere
    os: str
    device_class: str
    accelerator: str
    status: Status
    reason: str
    precision: str = ANY
    workload_type: str = ANY


SUPPORT_MATRIX: tuple[MatrixRow, ...] = (
    MatrixRow(
        ANY, ANY, ANY, ANY, ANY, "unsupported",
        "the iterative_action_policy (diffusion policy) workload is not yet supported for gating",
        workload_type="iterative_action_policy",
    ),
    MatrixRow(
        "tensorrt", None, "linux", "jetson", "none", "results_only",
        "TensorRT INT8 engines are calibrated on random noise, not representative data, so "
        "INT8 results are recorded but not gated until real-data calibration exists",
        precision="int8",
    ),
    MatrixRow(
        "tensorrt", None, "linux", "jetson", "none", "supported",
        "validated on Jetson Orin (clock pinning, telemetry, content-addressed engines)",
        precision="fp16",
    ),
    MatrixRow(
        "tensorrt", None, "linux", "jetson", "none", "supported",
        "validated on Jetson Orin (clock pinning, telemetry, content-addressed engines)",
        precision="fp32",
    ),
    MatrixRow(
        "onnxruntime", "cpu", "linux", "x86_64", "none", "results_only",
        "no x86 clock-pinning rule, and the hardware_family derivation not yet validated",
    ),
    MatrixRow(
        "onnxruntime", "cuda", "linux", "x86_64", "none", "results_only",
        "no x86 clock-pinning rule, and NVIDIA GPU clock state is not recorded",
    ),
    MatrixRow(
        "onnxruntime", "cuda", "linux", "jetson", "none", "results_only",
        "ONNX Runtime GPU path on Jetson not yet validated",
    ),
    MatrixRow(
        "tensorrt", None, "linux", "x86_64", "none", "results_only",
        "no x86 clock-pinning rule, and NVIDIA GPU clock state is not recorded",
    ),
    MatrixRow(
        "rknn", None, "linux", "rk3588", "none", "supported",
        "validated on the Orange Pi 5 Plus (declared clock state, the .rknn's build record "
        "binding it to its .onnx and conversion settings). FP16 output correctness is the "
        "customer's own quality evidence: the load-time canary is informational for FP16",
        precision="fp16",
    ),
    MatrixRow(
        "rknn", None, "linux", "rk3588", "none", "results_only",
        "the load-time correctness check can refuse a correct INT8 model, and INT8 output "
        "fidelity against the original model is not measured, so RKNN INT8 results are "
        "recorded but not gated",
        precision="int8",
    ),
    MatrixRow(
        "tflite", None, ANY, ANY, "none", "unsupported",
        "the TFLite adapter is not implemented",
    ),
)  # fmt: skip


def _matches(value: str | None, pattern: str | None) -> bool:
    return pattern == ANY or value == pattern


def lookup(
    runtime_family: str,
    execution_provider: str | None,
    os: str,
    device_class: str,
    accelerator: str = "none",
    *,
    precision: str | None = None,
    workload_type: str | None = None,
) -> MatrixRow | None:
    """The first row matching this combination, or None if the matrix has none.

    A row that names a precision or workload type matches only that value; an
    unknown (None) precision therefore never matches a precision-specific row.
    """
    for row in SUPPORT_MATRIX:
        if (
            _matches(runtime_family, row.runtime_family)
            and _matches(execution_provider, row.execution_provider)
            and _matches(os, row.os)
            and _matches(device_class, row.device_class)
            and _matches(accelerator, row.accelerator)
            and _matches(precision, row.precision)
            and _matches(workload_type, row.workload_type)
        ):
            return row
    return None


def status_of(row: MatrixRow | None) -> Status:
    """The row's status; a combination with no row is unsupported."""
    return row.status if row is not None else "unsupported"


def execution_provider_for(runtime_label: str) -> str | None:
    """ONNX Runtime's execution provider from its runtime label; None for other runtimes."""
    family, _, suffix = runtime_label.partition("-")
    return (suffix or None) if family == "onnxruntime" else None
