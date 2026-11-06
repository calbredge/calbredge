"""Unit tests for protocol/calbredge_protocol/graph_variant.py.

Two test classes:

- TestComputeGraphVariant / TestCompareVariantPair: fast, synthetic ONNX
  graphs built inline (same style as conftest.py's fixture synthesis) —
  pin the pure LOGIC of compute_graph_variant and compare_variant_pair with
  no dependency on real production model files. These run in the default
  CI pass.

- TestRealProductionModels: pins the ACTUAL verified 2026-09-08 result
  against the real Jetson and RK3588 source files — yolov8n/s/m/n_seg
  DIFFER (two variants), efficientnet_b0/efficientdet_lite0/mobilenetv2
  Jetson-vs-RK3588 are IDENTICAL in structure (one variant, despite
  different model_hash). Same skip-gracefully-if-absent pattern as
  tests/open/test_validate_rknn.py's ZOO_ONNX/STOCK_ONNX — these files live
  outside the repo (Jetson side: ~/calbredge-benchmarked-onnx-originals/,
  recovered from the Jetson device 2026-09-08 after the files that used to
  be at tests/open/fixtures/ for 3 of these models were found to no longer match
  what was actually benchmarked; RK3588 side: ~/calbredge-rknn-models/;
  this class skips on any machine without a copy).
"""

from pathlib import Path

import onnx
import pytest
from calbredge_protocol.graph_variant import (
    GraphSignature,
    compare_variant_pair,
    compute_graph_variant,
)
from onnx import TensorProto, helper

# ---------------------------------------------------------------------------
# Synthetic graph builders
# ---------------------------------------------------------------------------


def _build_onnx(tmp_path: Path, filename: str, *, weight_value: float, extra_output: bool) -> Path:
    """Build a minimal ONNX graph with a Conv + Relu, optionally a second
    output branch, and a configurable weight constant. Used to control
    exactly one structural property at a time across a pair of files, so
    each test isolates a single claim about compute_graph_variant.
    """
    conv_w = helper.make_tensor("conv_w", TensorProto.FLOAT, [1, 1, 1, 1], [weight_value])
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])]
    nodes = [
        helper.make_node("Conv", ["x", "conv_w"], ["conv_out"]),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
    ]
    outputs = [helper.make_tensor_value_info("relu_out", TensorProto.FLOAT, [1, 1, 4, 4])]

    if extra_output:
        nodes.append(helper.make_node("Relu", ["conv_out"], ["relu_out2"]))
        outputs.append(helper.make_tensor_value_info("relu_out2", TensorProto.FLOAT, [1, 1, 4, 4]))

    graph = helper.make_graph(
        nodes, "synthetic_graph_variant_test", inputs, outputs, initializer=[conv_w]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    dest = tmp_path / filename
    onnx.save(model, str(dest))
    return dest


# ---------------------------------------------------------------------------
# compute_graph_variant — pure logic, synthetic graphs
# ---------------------------------------------------------------------------


class TestComputeGraphVariant:
    @pytest.mark.unit
    def test_identical_structure_different_weights_same_variant(self, tmp_path):
        """The core claim behind treating efficientnet_b0/efficientdet_lite0
        as one variant: same node_count/output_shapes/op_types but different
        constant values (i.e. different file bytes, different model_hash)
        must produce the SAME graph_variant."""
        a = _build_onnx(tmp_path, "a.onnx", weight_value=1.0, extra_output=False)
        b = _build_onnx(tmp_path, "b.onnx", weight_value=999.0, extra_output=False)

        sig_a = compute_graph_variant(a)
        sig_b = compute_graph_variant(b)

        assert sig_a.graph_variant == sig_b.graph_variant
        assert sig_a.source_file_sha256 != sig_b.source_file_sha256  # different bytes, confirmed

    @pytest.mark.unit
    def test_different_output_count_different_variant(self, tmp_path):
        """The core claim behind splitting yolov8n/s/m/n_seg into two
        variants: an extra output tensor (the RKNN split-decode shape)
        must produce a DIFFERENT graph_variant, even with the weight held
        constant."""
        a = _build_onnx(tmp_path, "a.onnx", weight_value=1.0, extra_output=False)
        b = _build_onnx(tmp_path, "b.onnx", weight_value=1.0, extra_output=True)

        sig_a = compute_graph_variant(a)
        sig_b = compute_graph_variant(b)

        assert sig_a.graph_variant != sig_b.graph_variant
        assert sig_a.output_count == 1
        assert sig_b.output_count == 2

    @pytest.mark.unit
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_graph_variant(tmp_path / "does_not_exist.onnx")

    @pytest.mark.unit
    def test_signature_fields_populated(self, tmp_path):
        p = _build_onnx(tmp_path, "a.onnx", weight_value=1.0, extra_output=False)
        sig = compute_graph_variant(p)

        assert isinstance(sig, GraphSignature)
        assert sig.node_count == 2  # Conv + Relu
        assert sig.output_shapes == ("(1, 1, 4, 4)",)
        assert sig.op_histogram == {"Conv": 1, "Relu": 1}
        assert len(sig.source_file_sha256) == 64  # full hex SHA256
        assert len(sig.graph_variant) == 16


# ---------------------------------------------------------------------------
# compare_variant_pair
# ---------------------------------------------------------------------------


class TestCompareVariantPair:
    @pytest.mark.unit
    def test_fewer_outputs_classified_fused(self, tmp_path):
        a = _build_onnx(tmp_path, "a.onnx", weight_value=1.0, extra_output=False)  # 1 output
        b = _build_onnx(tmp_path, "b.onnx", weight_value=1.0, extra_output=True)  # 2 outputs
        sig_a = compute_graph_variant(a)
        sig_b = compute_graph_variant(b)

        assert compare_variant_pair(sig_a, sig_b) == ("fused", "split")
        assert compare_variant_pair(sig_b, sig_a) == ("split", "fused")  # order-independent

    @pytest.mark.unit
    def test_equal_output_count_raises(self, tmp_path):
        """Must not guess a fused/split label when there's no basis for
        one — e.g. two variants that differ only in node_count/op mix but
        not output_count."""
        a = _build_onnx(tmp_path, "a.onnx", weight_value=1.0, extra_output=False)
        b = _build_onnx(tmp_path, "b.onnx", weight_value=2.0, extra_output=False)
        sig_a = compute_graph_variant(a)
        sig_b = compute_graph_variant(b)

        with pytest.raises(ValueError, match="Cannot classify fused/split"):
            compare_variant_pair(sig_a, sig_b)


# ---------------------------------------------------------------------------
# Real production models — pins the actual 2026-09-08 verified result
# ---------------------------------------------------------------------------

# Jetson-side: recovered from the Jetson device 2026-09-08 and independently
# hash-verified against live model_registry.model_hash (see
# MANIFEST.sha256 in the same directory). NOT tests/open/fixtures/ — 3 of that
# directory's files (yolov8s, efficientnet_b0, efficientdet_lite0) were
# discovered 2026-09-08 to no longer match what was actually benchmarked
# (scripts/download_models.py re-downloads/re-exports and overwrote them at
# some point after the historical benchmark uploads).
_JETSON_DIR = Path.home() / "calbredge-benchmarked-onnx-originals"

# RK3588-side: the RKNN conversion host's model cache. This class skips
# gracefully on a machine without a copy.
_RK_DIR = Path.home() / "calbredge-rknn-models"

# (jetson filename, rk3588 filename, expect_same_variant)
_REAL_PAIRS = [
    ("yolov8n.onnx", "yolov8n.onnx", False),
    ("yolov8s.onnx", "yolov8s.onnx", False),
    ("yolov8m.onnx", "yolov8m.onnx", False),
    # RK3588-side filename uses a hyphen, not the model_label's underscore —
    # confirmed against model_registry, not a guess.
    ("yolov8n_seg.onnx", "yolov8n-seg.onnx", False),
    ("efficientnet_b0.onnx", "efficientnet_b0.onnx", True),
    ("efficientdet_lite0.onnx", "efficientdet_lite0.onnx", True),
    ("mobilenetv2.onnx", "mobilenetv2.onnx", True),
]

_REAL_FILES_PRESENT = (
    _JETSON_DIR.is_dir()
    and _RK_DIR.is_dir()
    and all((_JETSON_DIR / j).exists() and (_RK_DIR / r).exists() for j, r, _ in _REAL_PAIRS)
)


@pytest.mark.integration
@pytest.mark.skipif(
    not _REAL_FILES_PRESENT,
    reason=(
        f"requires the real production ONNX files "
        f"(jetson_dir={_JETSON_DIR.is_dir()}, rk3588_dir={_RK_DIR.is_dir()})"
    ),
)
class TestRealProductionModels:
    @pytest.mark.parametrize("jetson_file,rk3588_file,expect_same", _REAL_PAIRS)
    def test_jetson_vs_rk3588_variant(self, jetson_file, rk3588_file, expect_same):
        sig_j = compute_graph_variant(_JETSON_DIR / jetson_file)
        sig_r = compute_graph_variant(_RK_DIR / rk3588_file)

        same = sig_j.graph_variant == sig_r.graph_variant
        assert same == expect_same, (
            f"{jetson_file}: expected same_variant={expect_same}, got {same} "
            f"(jetson: {sig_j.node_count} nodes/{sig_j.output_count} outputs, "
            f"rk3588: {sig_r.node_count} nodes/{sig_r.output_count} outputs)"
        )

    def test_yolov8n_decode_location_split_correctly(self):
        """Spot-check compare_variant_pair against the one pair whose real
        node/output counts are already known: Jetson 249
        nodes/1 output (fused), RK3588 228 nodes/9 outputs (split)."""
        sig_j = compute_graph_variant(_JETSON_DIR / "yolov8n.onnx")
        sig_r = compute_graph_variant(_RK_DIR / "yolov8n.onnx")

        assert (sig_j.node_count, sig_j.output_count) == (249, 1)
        assert (sig_r.node_count, sig_r.output_count) == (228, 9)
        assert compare_variant_pair(sig_j, sig_r) == ("fused", "split")
