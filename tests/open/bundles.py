"""Run bundles for tests: a TensorRT-on-Jetson v2 bundle built from the golden result."""

import json
from pathlib import Path

from calbredge.hasher import compute_model_hash
from calbredge_protocol.stats import compute_latency_stats

GOLDEN_RESULT = Path(__file__).parent / "fixtures" / "v2" / "golden_result_v2_0.json"
FAMILY = "nvidia_jetson_orin_nano_engineering_reference_developer_kit_super"
ENGINE_BUILD = {
    "build_key": "c" * 64,
    "built_from_model_hash": None,  # set per model below
    "tensorrt_version": "10.3.0",
    "calibration_digest": None,
}


def make_bundle(root: Path, mutate=None, machine: str = "aarch64") -> Path:
    """A TensorRT-on-Jetson v2 bundle whose statistics come from its samples.

    Schema 2.1 with an engine build record, as current code writes: a bundle
    from before content-addressed engines is refused (legacy_reasons).
    """
    data = json.loads(GOLDEN_RESULT.read_text())
    data["schema_version"] = "2.1"
    data["candidate"]["engine_build"] = {
        "build_key": "c" * 64,
        "built_from_model_hash": data["candidate"]["model_hash"],
        "tensorrt_version": "10.3.0",
        "calibration_digest": None,
    }
    samples = [4.40 + 0.001 * (i % 37) for i in range(500)]
    stats = compute_latency_stats(samples)
    data["metrics"].update(
        latency_p50_ms=stats.p50_ms,
        latency_p90_ms=stats.p90_ms,
        latency_p99_ms=stats.p99_ms,
        latency_mean_ms=stats.mean_ms,
        latency_std_ms=stats.std_ms,
        sample_count=stats.sample_count,
    )
    if mutate:
        mutate(data)
    bundle = root / data["result_id"]
    bundle.mkdir(parents=True)
    (bundle / "result.json").write_text(json.dumps(data, indent=2))
    (bundle / "samples.jsonl").write_text(
        "".join(json.dumps({"iteration": i, "latency_ms": s}) + "\n" for i, s in enumerate(samples))
    )
    (bundle / "environment.json").write_text(json.dumps({"platform": {"machine": machine}}))
    return bundle


def _bundle(root: Path, model: Path, n: int, timestamp: str, **changes) -> Path:
    """An eligible (post content-addressed engines) bundle for this model, unless changed."""

    def mutate(d):
        d["result_id"] = f"00000000-0000-4000-8000-{n:012d}"
        d["timestamp"] = timestamp
        d["candidate"]["model_hash"] = compute_model_hash(model)
        d["candidate"]["engine_build"] = {
            **ENGINE_BUILD,
            "built_from_model_hash": compute_model_hash(model),
        }
        for section, values in changes.items():
            if section == "schema_version":
                d["schema_version"] = values
            else:
                d[section].update(values)

    return make_bundle(root, mutate)
