"""Computes a structural signature for an ONNX graph.

Exists to distinguish genuinely different graph exports of "the same" model
(e.g. Jetson's stock Ultralytics export of yolov8n, which fuses detection
decode into a single (1,84,8400) output tensor, vs the Rockchip-zoo export
used for RK3588/RKNN, which moves decode off-NPU and ships 9 per-scale
tensors instead) from exports that merely differ in file bytes but not in
graph shape (confirmed for efficientnet_b0 and efficientdet_lite0: their
Jetson and RK3588 source files have different SHA256 — hence different
model_registry.model_hash rows — but identical node_count, output_shape set,
and op_type histogram; investigated 2026-09-08).

model_hash (SHA256 of file bytes, see agent/calbredge/hasher.py) already
answers "is this the exact same file" — it does not answer "is this the same
computational graph shape," which is the question that actually determines
whether two runs are comparable. graph_variant answers that second question,
computed once per distinct model file and stored on model_registry.

Deliberately does NOT use benchmark_runs.input_shape as a signature input —
confirmed (2026-09-08, via a same-model_hash control: cxr_mobilenetv2, which
has a single model_registry row and only one real source file) that column
is adapter-resolved at run time, not read from the graph: the RKNN adapter
reports its own NCHW convention regardless of the file's own declared input
layout, so the same exact file produces two different input_shape values
depending on which runtime ran it. Using it here would falsely split a
single real model into two variants.

IMPORTANT — graph_variant is NOT a global model identity or a unique key.
It is only meaningful scoped to one model_label. Confirmed 2026-09-08:
yolov8n's and yolov8s' RK3588 rows compute the exact same graph_variant,
because YOLOv8 n/s share graph topology (channel width scales tensor sizes,
not node_count or op_type counts) — two DIFFERENT models can legitimately
share a graph_variant. Any comparison or grouping must use the pair
(model_label, graph_variant), never graph_variant alone.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import onnx

# Matches agent/calbredge/hasher.py's chunk size — same rationale (flat,
# predictable memory use regardless of model file size; production ONNX
# exports here run 6-100+ MB).
_HASH_CHUNK_SIZE = 8192


@dataclass(frozen=True)
class GraphSignature:
    """Structural facts about one ONNX graph, plus the derived signature.

    Instances are frozen — a signature describes one specific file's graph
    shape at the moment it was computed and should not be mutated in place.
    """

    graph_variant: str
    """Short (16 hex char) hash of (node_count, sorted output_shapes,
    sorted op_type histogram). Two files with this same value have the same
    graph shape regardless of whether their model_hash (file bytes) match —
    this is the equality/grouping key.

    NOT a global model identity or unique key — scoped to one model_label
    only. Two different models can share a graph_variant (confirmed:
    yolov8n and yolov8s' RK3588 exports do, since n/s share topology).
    Always group/compare as (model_label, graph_variant), never
    graph_variant alone."""

    node_count: int
    """Number of nodes in the ONNX graph (len(graph.node))."""

    output_shapes: tuple[str, ...]
    """Every graph output's shape, as strings (e.g. "(1, 84, 8400)"; a
    symbolic dim shows as "?"), sorted — order in the ONNX file is not
    semantically meaningful and must not affect the signature."""

    op_histogram: dict[str, int]
    """Count of each node op_type in the graph (e.g. {"Conv": 64, ...}).
    Included alongside node_count so two graphs with the same node COUNT but
    a genuinely different mix of operations are still told apart."""

    source_file_sha256: str
    """Full SHA256 of the file this signature was computed from — same
    value as agent/calbredge/hasher.py's compute_model_hash would produce
    for the identical file, included here so graph_meta can state which
    exact file backed a given signature without a second hashing pass."""

    @property
    def output_count(self) -> int:
        """Number of graph outputs. Convenience accessor — same information
        as len(output_shapes), named to match graph_meta's output_count
        field."""
        return len(self.output_shapes)


def compute_graph_variant(onnx_path: str | Path) -> GraphSignature:
    """Load an ONNX file and compute its structural signature.

    Reads only the graph's node list, output shapes, and op types — never
    weight values, so two files with identical topology but different
    trained weights (or re-export timestamp/metadata noise, the confirmed
    efficientnet_b0/efficientdet_lite0 case) produce the same graph_variant.
    Uses load_external_data=False: this function only inspects graph
    structure, never needs weight tensors, and skipping them avoids an
    unnecessary read of large external-data files some ONNX exports use.

    Args:
        onnx_path: Path to a ``.onnx`` file.

    Returns:
        A GraphSignature describing the file's graph shape.

    Raises:
        FileNotFoundError: If onnx_path does not exist.
        Exception: Whatever onnx.load raises for a malformed/unreadable file
                   — not caught here; a corrupt model file should fail loud,
                   not silently produce a signature for nothing.
    """
    model = load_graph(onnx_path)
    return signature_from_model(model, _sha256_file(Path(onnx_path)))


def load_graph(onnx_path: str | Path) -> onnx.ModelProto:
    """Load an ONNX file's graph structure without its external weight data.

    Shared by compute_graph_variant and the v2 contract derivation
    (agent/calbredge/onnx_contract.py) so a large model is read once and with
    the same options everywhere.

    Raises:
        FileNotFoundError: If onnx_path does not exist.
        Exception: Whatever onnx.load raises for a malformed/unreadable file.
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    return onnx.load(str(path), load_external_data=False)


def signature_from_model(model: onnx.ModelProto, source_file_sha256: str) -> GraphSignature:
    """Compute the GraphSignature of an already-loaded model."""
    graph = model.graph

    node_count = len(graph.node)

    output_shapes = tuple(
        sorted(
            str(
                tuple(
                    d.dim_value if d.dim_value else (d.dim_param or "?")
                    for d in output.type.tensor_type.shape.dim
                )
            )
            for output in graph.output
        )
    )

    op_histogram = dict(Counter(node.op_type for node in graph.node))

    # json.dumps(sort_keys=True) over a plain list, not a dict, because the
    # top-level structure is positional (node_count, output_shapes,
    # op_histogram) — sort_keys only needs to make op_histogram's own key
    # order deterministic, which it does since op_histogram is the dict
    # passed inside the list.
    signature_input = json.dumps(
        [node_count, list(output_shapes), op_histogram],
        sort_keys=True,
    )
    graph_variant = hashlib.sha256(signature_input.encode("utf-8")).hexdigest()[:16]

    return GraphSignature(
        graph_variant=graph_variant,
        node_count=node_count,
        output_shapes=output_shapes,
        op_histogram=op_histogram,
        source_file_sha256=source_file_sha256,
    )


def _sha256_file(path: Path) -> str:
    """Chunked SHA256 of a file's bytes. Same algorithm as
    agent/calbredge/hasher.py's compute_model_hash — not imported from
    there directly to keep protocol/ free of a dependency on agent/
    (protocol is the shared engine; agent depends on it, not the reverse)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def compare_variant_pair(a: GraphSignature, b: GraphSignature) -> tuple[str, str]:
    """Classify which of two same-model_label GraphSignatures is the more
    "decode-fused" export and which is "decode-split", for graph_meta's
    decode_location field.

    Classifies purely by output_count — confirmed against all 4 real
    fused/split pairs found in production data (yolov8n, yolov8s, yolov8m,
    yolov8n_seg: Jetson stock always 1-2 outputs, RK3588 zoo always 9-13)
    that output_count is the property the domain actually means by
    "fused vs split" (a single decode tensor vs decode moved off-graph into
    per-scale tensors) — node_count is NOT used for this classification: it
    moves in the OPPOSITE direction (RK3588's node_count is LOWER than
    Jetson's for all 4 pairs, since removing the fused decode subgraph
    removes nodes, even though output_count goes up), so a node_count-based
    rule would get every one of the 4 real pairs backwards.

    Only meaningful for two signatures that share a model_label but have
    different graph_variant — the caller is responsible for that pairing;
    this function only compares the two GraphSignatures given.

    Args:
        a: First variant's signature.
        b: Second variant's signature.

    Returns:
        (a_label, b_label) where each label is "fused" or "split".

    Raises:
        ValueError: If a.output_count == b.output_count — not classifiable
                    by this rule, and must not be guessed. Callers should
                    leave decode_location null for such a pair rather than
                    invent an answer.
    """
    if a.output_count < b.output_count:
        return "fused", "split"
    if b.output_count < a.output_count:
        return "split", "fused"
    raise ValueError(
        f"Cannot classify fused/split: both variants have "
        f"output_count={a.output_count}. This pair needs manual review, "
        "not a guessed decode_location."
    )
