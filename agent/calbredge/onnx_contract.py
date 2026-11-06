"""Derive the v2 input/output contract from a source .onnx model.

Implements docs/RESULT_SCHEMA.md §3.3.1 (precision for runtimes without a
precision suffix) and §3.5 (io_signature, input_shape, graph_kind, and what
makes a graph data_dependent).

Parsing a large model is expensive and, on memory-fragile Jetsons, risky (the
961 MB diffusion UNet), so the graph-intrinsic results are cached per
model_hash in a sidecar file next to the model and parsed once. The cache
never holds anything that depends on the protocol file: input_shape's
dynamic dimensions are bound per call.
"""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import onnx
from calbredge_protocol.graph_variant import load_graph

# Bump whenever a derivation rule below changes: cached results carrying an
# older version are ignored and recomputed.
DERIVATION_VERSION = 1

SIDECAR_SUFFIX = ".calbredge-contract.json"

# Ops whose amount of work depends on input values (RESULT_SCHEMA.md §3.5).
DATA_DEPENDENT_OPS = frozenset(
    {"NonMaxSuppression", "If", "Loop", "Scan", "NonZero", "Unique", "Compress"}
)
# Ops that mean the graph computes in int8 even though its inputs are float.
_INT8_OPS = frozenset(
    {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "MatMulInteger", "ConvInteger"}
)
_WEIGHT_DTYPE_TO_PRECISION: dict[int, str] = {
    onnx.TensorProto.FLOAT: "fp32",
    onnx.TensorProto.FLOAT16: "fp16",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.UINT8: "int8",
}


@dataclass(frozen=True)
class GraphContract:
    """Everything derivable from the graph alone, independent of any protocol."""

    io_signature: dict[str, Any]
    primary_input_dims: list[int | str]
    graph_kind: str
    graph_kind_triggers: list[str]
    graph_precision: str | None


def derive_graph_contract(model: onnx.ModelProto) -> GraphContract:
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}
    nodes = list(_all_nodes(graph))
    constant_names = initializer_names | {
        out for n in nodes if n.op_type == "Constant" for out in n.output
    }
    for sub in _subgraphs(graph):
        constant_names |= {t.name for t in sub.initializer}

    inputs = [_tensor_spec(v) for v in graph.input if v.name not in initializer_names]
    outputs = [_tensor_spec(v) for v in graph.output]
    if not inputs:
        raise ValueError("model has no non-initializer inputs")

    triggers = sorted({n.op_type for n in nodes if n.op_type in DATA_DEPENDENT_OPS})
    for n in nodes:
        # TopK with a constant K always selects the same number of elements;
        # a K computed at run time makes the downstream work data-dependent.
        if n.op_type == "TopK" and (len(n.input) < 2 or n.input[1] not in constant_names):
            triggers.append("TopK(non-constant K)")
    return GraphContract(
        io_signature={"inputs": inputs, "outputs": outputs},
        primary_input_dims=list(inputs[0]["shape"]),
        graph_kind="data_dependent" if triggers else "fixed_compute",
        graph_kind_triggers=sorted(set(triggers)),
        graph_precision=_graph_precision(graph, nodes),
    )


def bind_input_shape(
    primary_input_dims: list[int | str], protocol_input_shape: list[int]
) -> list[int]:
    """Static dims come from the graph; dynamic (named) dims from the protocol."""
    if len(primary_input_dims) != len(protocol_input_shape):
        raise ValueError(
            f"model's primary input has {len(primary_input_dims)} dimensions "
            f"{primary_input_dims}, protocol input_shape has {len(protocol_input_shape)} "
            f"{protocol_input_shape}"
        )
    return [
        d if isinstance(d, int) else p
        for d, p in zip(primary_input_dims, protocol_input_shape, strict=True)
    ]


def load_graph_contract(
    model_path: Path, model_hash: str, fallback_cache_dir: Path | None = None
) -> tuple[GraphContract, str]:
    """Return the model's GraphContract, parsing the model only on a cache miss.

    Looks for ``<model>.onnx.calbredge-contract.json`` next to the model, then
    ``<fallback_cache_dir>/<model_hash>.json``. A cache entry is used only if
    both its model_hash and DERIVATION_VERSION match. On a miss the contract
    is derived and written to the sidecar, or to the fallback dir when the
    model's directory is read-only; failing to write never fails the call.

    Returns (contract, cache_status) where cache_status is "hit:<path>",
    "written:<path>" or "unwritable".
    """
    candidates = [sidecar_path(model_path)]
    if fallback_cache_dir is not None:
        candidates.append(fallback_cache_dir / f"{model_hash}.json")
    for path in candidates:
        cached = _read_cache(path, model_hash)
        if cached is not None:
            return cached, f"hit:{path}"

    contract = derive_graph_contract(load_graph(model_path))
    payload = {
        "derivation_version": DERIVATION_VERSION,
        "model_hash": model_hash,
        "contract": asdict(contract),
    }
    for path in candidates:
        if _write_atomic(path, payload):
            return contract, f"written:{path}"
    return contract, "unwritable"


def sidecar_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + SIDECAR_SUFFIX)


# --- internals ---------------------------------------------------------------


def _subgraphs(graph: onnx.GraphProto):
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield attr.g
                yield from _subgraphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield g
                    yield from _subgraphs(g)


def _all_nodes(graph: onnx.GraphProto):
    yield from graph.node
    for sub in _subgraphs(graph):
        yield from sub.node


def _dtype_name(elem_type: int) -> str:
    try:
        return onnx.helper.tensor_dtype_to_np_dtype(elem_type).name
    except (KeyError, TypeError, ValueError):
        return onnx.TensorProto.DataType.Name(elem_type).lower()


def _tensor_spec(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape: list[int | str] = [
        d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
        for d in tensor_type.shape.dim
    ]
    return {"name": value.name, "dtype": _dtype_name(tensor_type.elem_type), "shape": shape}


def _graph_precision(graph: onnx.GraphProto, nodes: list[onnx.NodeProto]) -> str | None:
    if any(n.op_type in _INT8_OPS or n.op_type.startswith("QLinear") for n in nodes):
        return "int8"
    elements: dict[int, int] = {}
    tensors = list(graph.initializer) + [
        a.t for n in nodes if n.op_type == "Constant" for a in n.attribute if a.name == "value"
    ]
    for sub in _subgraphs(graph):
        tensors += list(sub.initializer)
    for t in tensors:
        elements[t.data_type] = elements.get(t.data_type, 0) + math.prod(t.dims)
    if not elements:
        return None
    dominant = max(elements, key=lambda k: elements[k])
    return _WEIGHT_DTYPE_TO_PRECISION.get(dominant)


def _read_cache(path: Path, model_hash: str) -> GraphContract | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("derivation_version") != DERIVATION_VERSION
            or payload.get("model_hash") != model_hash
        ):
            return None
        return GraphContract(**payload["contract"])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _write_atomic(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
