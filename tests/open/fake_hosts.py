"""Fake host filesystem trees and tiny ONNX models for the v2 collector tests.

The Jetson values are the real unprivileged readings from the Orin Nano Super
(JetPack 6, L4T R36.4.7) with clocks pinned by jetson_clocks and after a
reboot (unpinned), so the tests exercise the exact shapes the collectors see.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ORIN_MODEL = "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"
ORIN_FAMILY = "nvidia_jetson_orin_nano_engineering_reference_developer_kit_super"
ORIN_COMPATIBLE = ["nvidia,p3768-0000+p3767-0005-super", "nvidia,p3767-0005", "nvidia,tegra234"]
L4T_RELEASE = (
    "# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, "
    "DATE: Thu Sep 18 22:54:44 UTC 2025\n"
)
CUDA_VERSION_JSON = (
    '{\n   "cuda" : {\n      "name" : "CUDA SDK",\n      "version" : "12.6.11"\n   }\n}\n'
)


def _write(root: Path, rel: str, content: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def _device_tree(root: Path, model: str, compatible: list[str]) -> None:
    _write(root, "proc/device-tree/model", model.encode() + b"\x00")
    _write(root, "proc/device-tree/compatible", b"".join(c.encode() + b"\x00" for c in compatible))


def build_jetson_orin(root: Path, *, pinned: bool, gpu_node: str = "17000000.gpu") -> Path:
    _device_tree(root, ORIN_MODEL, ORIN_COMPATIBLE)
    cpu_min = "1728000" if pinned else "729600"
    for policy, cpus in (("policy0", "0 1"), ("policy2", "2 3 4 5")):
        base = f"sys/devices/system/cpu/cpufreq/{policy}"
        _write(root, f"{base}/related_cpus", cpus + "\n")
        _write(root, f"{base}/scaling_min_freq", cpu_min + "\n")
        _write(root, f"{base}/scaling_max_freq", "1728000\n")
        _write(root, f"{base}/scaling_governor", "schedutil\n")
    gpu_min = "1020000000" if pinned else "306000000"
    gpu = f"sys/class/devfreq/{gpu_node}"
    _write(root, f"{gpu}/min_freq", gpu_min + "\n")
    _write(root, f"{gpu}/max_freq", "1020000000\n")
    _write(root, f"{gpu}/cur_freq", gpu_min + "\n")
    _write(root, f"{gpu}/governor", "nvhost_podgov\n")
    # Non-GPU devfreq nodes stay min < max even when pinned; they must be ignored.
    _write(root, "sys/class/devfreq/15340000.vic/min_freq", "115200000\n")
    _write(root, "sys/class/devfreq/15340000.vic/max_freq", "435200000\n")
    _write(root, "etc/nv_tegra_release", L4T_RELEASE)
    _write(root, "usr/local/cuda/version.json", CUDA_VERSION_JSON)
    # BPMP debugfs deliberately absent: unprivileged reads are denied.
    return root


# kHz, as the Orange Pi 5 Plus reports them: A55 x4, A76 x2, A76 x2.
RK3588_CPU_MAX_KHZ = {"policy0": 1800000, "policy4": 2400000, "policy6": 2400000}
RK3588_DEVFREQ_MAX_HZ = {"fdab0000.npu": 1000000000, "dmc": 2112000000, "fb000000.gpu": 1000000000}


def build_rk3588(
    root: Path,
    *,
    governor: str = "performance",
    gpu_governor: str | None = "simple_ondemand",
    scaling_max_khz: dict[str, int] | None = None,
) -> Path:
    """An Orange Pi 5 Plus after scripts/performance_mode.sh: CPU, NPU and DMC at ``governor``
    (rknpu_ondemand and dmc_ondemand when it is ``ondemand``), the GPU left at its default.
    ``scaling_max_khz`` lowers a CPU policy's scaling maximum below its hardware maximum."""
    _device_tree(root, "Orange Pi 5 Plus", ["rockchip,rk3588-orangepi-5-plus", "rockchip,rk3588"])
    pinned = governor == "performance"
    for policy, hardware_max in RK3588_CPU_MAX_KHZ.items():
        base = f"sys/devices/system/cpu/cpufreq/{policy}"
        scaling_max = (scaling_max_khz or {}).get(policy, hardware_max)
        _write(root, f"{base}/scaling_governor", governor + "\n")
        _write(root, f"{base}/scaling_cur_freq", f"{scaling_max if pinned else 408000}\n")
        _write(root, f"{base}/scaling_min_freq", "408000\n")
        _write(root, f"{base}/scaling_max_freq", f"{scaling_max}\n")
        _write(root, f"{base}/cpuinfo_max_freq", f"{hardware_max}\n")
    devfreq = {
        "fdab0000.npu": governor if pinned else "rknpu_ondemand",
        "dmc": governor if pinned else "dmc_ondemand",
        "fb000000.gpu": gpu_governor,
    }
    for node, node_governor in devfreq.items():
        if node_governor is None:
            continue
        hardware_max = RK3588_DEVFREQ_MAX_HZ[node]
        _write(root, f"sys/class/devfreq/{node}/governor", node_governor + "\n")
        _write(root, f"sys/class/devfreq/{node}/cur_freq", f"{hardware_max}\n")
        _write(root, f"sys/class/devfreq/{node}/max_freq", f"{hardware_max}\n")
    return root


def build_x86(root: Path, *, governor: str = "powersave") -> Path:
    for policy in ("policy0", "policy1"):
        _write(root, f"sys/devices/system/cpu/cpufreq/{policy}/scaling_governor", governor + "\n")
    return root


# --- tiny ONNX models --------------------------------------------------------------


# Pinned so the models load in whichever onnxruntime CI installs: newer onnx
# releases default to an IR version older runtimes reject ("Unsupported model
# IR version: 14, max supported IR version: 13"). Opset 17 needs IR >= 8.
_IR_VERSION = 8


def _save(model: onnx.ModelProto, path: Path) -> Path:
    model.ir_version = _IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def conv_model(path: Path, *, fp16_weights: bool = False, dynamic_batch: bool = True) -> Path:
    """Input [N,3,8,8] -> Conv -> output. fixed_compute."""
    elem = TensorProto.FLOAT16 if fp16_weights else TensorProto.FLOAT
    weight_dtype = np.float16 if fp16_weights else np.float32
    batch: int | str = "N" if dynamic_batch else 1
    w = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=weight_dtype), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"])],
        "conv",
        [helper.make_tensor_value_info("x", elem, [batch, 3, 8, 8])],
        [helper.make_tensor_value_info("y", elem, [batch, 4, 6, 6])],
        initializer=[w],
    )
    return _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def nms_model(path: Path) -> Path:
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 10, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 10])
    out = helper.make_tensor_value_info("selected", TensorProto.INT64, ["K", 3])
    graph = helper.make_graph(
        [helper.make_node("NonMaxSuppression", ["boxes", "scores"], ["selected"])],
        "nms",
        [boxes, scores],
        [out],
    )
    return _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def if_model(path: Path, *, nested_nms: bool = False) -> Path:
    """An If whose branches are subgraphs; optionally NMS inside a branch."""
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    const = helper.make_tensor("c", TensorProto.FLOAT, [1], [1.0])
    then_nodes = [helper.make_node("Constant", [], ["t"], value=const)]
    if nested_nms:
        then_nodes.append(helper.make_node("NonZero", ["t"], ["nz"]))
    then_g = helper.make_graph(
        then_nodes, "then", [], [helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])]
    )
    else_g = helper.make_graph(
        [helper.make_node("Constant", [], ["e"], value=const)],
        "else",
        [],
        [helper.make_tensor_value_info("e", TensorProto.FLOAT, [1])],
    )
    graph = helper.make_graph(
        [helper.make_node("If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g)],
        "if",
        [cond],
        [y],
    )
    return _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def topk_model(path: Path, *, constant_k: bool) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 100])
    outs = [
        helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, "k"]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [1, "k"]),
    ]
    inputs = [x]
    initializers = []
    if constant_k:
        initializers.append(numpy_helper.from_array(np.array([5], dtype=np.int64), "k"))
    else:
        inputs.append(helper.make_tensor_value_info("k", TensorProto.INT64, [1]))
    graph = helper.make_graph(
        [helper.make_node("TopK", ["x", "k"], ["values", "indices"])],
        "topk",
        inputs,
        outs,
        initializer=initializers,
    )
    return _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def qdq_model(path: Path) -> Path:
    """Float input and output, int8 inside: QuantizeLinear -> DequantizeLinear."""
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.array(0, dtype=np.int8), "zero")
    big = numpy_helper.from_array(np.ones((64, 64), dtype=np.float32), "unused_float_weight")
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["y"]),
        ],
        "qdq",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[scale, zero, big],
    )
    return _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
