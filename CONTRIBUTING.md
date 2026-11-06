# Contributing to Calbredge

Thank you for helping. This guide covers reporting a problem, proposing a change, setting up a
development checkout, and the rules for changes that can affect what Calbredge measures.

## Reporting a problem

Issues are welcome. There is no support commitment, so a reply or a fix may take a while.

A bug report is most useful with:

- the board (for example a Jetson Orin Nano Super, or an RK3588 board);
- the runtime and precision (for example `tensorrt-fp16` or `rknn-fp16`);
- the Calbredge version, from `calbredge --version`;
- the output of `calbredge doctor`;
- the output of `calbredge inspect --model <your model>.onnx --runtime <runtime>`.

Report a security problem privately, through GitHub's private vulnerability reporting (the
repository's Security tab), never in a public issue.

## Proposing a change

Open an issue describing the change before writing a pull request, so the change can be agreed
first. For now this repository is updated by snapshot: an accepted change is applied in the
maintainer's repository, with credit to its author, and appears here in the next update.

## Setting up

From a clone, in a virtual environment, install both packages editable and the development
tools, as CI does:

```bash
python3 -m venv ~/calbredge-dev
source ~/calbredge-dev/bin/activate
python -m pip install --upgrade pip
pip install -e ./protocol
pip install -e "./agent[onnxruntime,rknn-convert]"
pip install -r dev-requirements.txt
```

Then run the checks as `.github/workflows/ci.yml` runs them:

```bash
pytest -q --cov --cov-report=term --cov-fail-under=89
ruff check .
ruff format --check .
pyright --outputjson agent protocol scripts > pyright-report.json
python scripts/check_pyright_ratchet.py --counts pyright-counts.json --report pyright-report.json
```

`pyright-counts.json` records how many pyright errors each file has; the last command fails
when a file has more than recorded. pyright itself exits non-zero while any error remains,
so read the result from the ratchet. When a change removes errors, record the fall with
`--update` on the same command.

To run ruff on every commit, install the hooks once (pre-commit is not in
`dev-requirements.txt`):

```bash
pip install pre-commit
pre-commit install
```

Setting up a target to measure on: [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Measurement integrity

Calbredge is only useful if its numbers can be trusted. A change that looks harmless, such as
moving a synchronization call, generating the input inside the timed loop, or recording 0 where
a value is missing, can shift every result without failing a test, and runs measured before
and after it then stop being comparable. The files below decide what is measured, whether two
runs compare, and how results are validated, so a change to them is held to a higher bar.

### Protected files

- `protocol/calbredge_protocol/tensorrt_adapter.py`: the timed loop and stream sync (`run_benchmark`, `warmup`, `run_inference`), buffer allocation and input handling, engine build, INT8 calibration
- `protocol/calbredge_protocol/rknn_adapter.py`: the timed loop and power-meter bracket (`run_benchmark`), inputs, `.rknn` lookup, the load-time canary, telemetry, governor collection
- `protocol/calbredge_protocol/onnx_adapter.py`: the timed loop and CUDA sync, input preparation, provider selection
- `protocol/calbredge_protocol/adapters.py`: the base timed loop, the post-loop tegrastats snapshot (`collect_hardware_metrics`), `run_status` thresholds (`classify_run`), the power-mode label
- `protocol/calbredge_protocol/power_meter.py`: the RK3588 external meter's window and energy
- `protocol/calbredge_protocol/power_window.py`: the windowed tegrastats sampler
- `protocol/calbredge_protocol/stats.py`: what p50, p90 and p99 mean (`compute_latency_stats`)
- `protocol/calbredge_protocol/throughput.py`: throughput from p50
- `protocol/calbredge_protocol/loader.py`: protocol parsing
- `protocol/calbredge_protocol/models.py`: `BenchmarkProtocol` and its defaults, `protocol_digest`, `HardwareMetrics`
- `protocol/calbredge_protocol/protocols/*.yaml`: the bundled protocols (iterations, input shapes)
- `protocol/calbredge_protocol/engine_cache.py`: which compiled engine is reused (build key, record check)
- `protocol/calbredge_protocol/rknn_artifact.py`: the `.rknn` build record (build key, artifact hash, record check)
- `protocol/calbredge_protocol/derive_v2.py`: `clocks_pinned`, `cpu_affinity_tier`, the derived hardware family
- `protocol/calbredge_protocol/iterative_policy.py`: the `iterative_action_policy` timing
- `protocol/calbredge_protocol/result_v2.py`: v2 result validation
- `protocol/calbredge_protocol/canonical.py`: canonical JSON and digests
- `protocol/calbredge_protocol/support_matrix.py`: which combinations are supported, results only or unsupported
- `agent/calbredge/cli.py`: `_execute_benchmark` (affinity before load, input generation, the load, warm-up and measure order), the legacy stability label
- `agent/calbredge/affinity.py`: CPU affinity detection and pinning
- `agent/calbredge/clock_state.py`: clock-state collection (the performance state)
- `agent/calbredge/collector.py`: system metadata, the runtime version
- `agent/calbredge/fingerprint.py`: `hardware_id`
- `agent/calbredge/host_v2.py`: `hardware_family`, `runtime_version`, `power_mode`, host classification
- `agent/calbredge/hasher.py`: model identity
- `agent/calbredge/onnx_contract.py`: the I/O signature, `graph_kind` and graph precision
- `agent/calbredge/result_builder.py`: what a v2 result records
- `agent/calbredge/bundle.py`: what a run bundle holds (`samples.jsonl`, `result.json`)
- `examples/systemd/calbredge-jetson-clocks.service`: the Jetson's clocks at boot
- `examples/systemd/calbredge-rk3588-clocks.service`: the RK3588's clocks at boot
- `scripts/performance_mode.sh`: the RK3588 governors
- `tests/open/fixtures/v1/golden_upload_mock.json`: the golden v1 upload payload
- `tests/open/fixtures/v2/golden_result_v2_0.json`: the golden v2 result
- `tests/open/test_tensorrt_adapter.py`: the TensorRT adapter's timing guards
- `tests/open/test_rknn_adapter.py`: the RKNN adapter's timing guards

### The invariants

- The device is synchronized before the timer stops, with both copies inside the timed region.
- Warm-up is never timed; the timed loop measures only inference.
- A missing value is null, never 0.
- No silent substitution: a fallback is recorded or refused, never hidden.
- Record what ran: runtime version, engine build key, clocks, power mode, protocol digest,
  telemetry boundary.

### What a change to a protected file includes

- **One purpose per change.** A change to a protected file is its own pull request, with no
  refactors, renames or reformatting alongside it.
- **A statement of its measurement impact**, as a `Measurement-Impact:` line in the pull
  request description:
  - `none` only when the change cannot alter a recorded value or how results are validated
    (docstrings, log text, comments, test-only seams). When in doubt, it is `intended`.
  - `intended`, with a `Measurement-Evidence:` line saying what shows the change.
- **For a change to a timed loop or to what is recorded, before-and-after runs** on each
  target the change can reach: a Jetson (TensorRT FP16) and an RK3588 (RKNN FP16), at least 3
  runs each side for each model. A change that does not depend on the model (the timed loop,
  the result path, the CLI) is shown with yolov8n and the example MobileNetV2; a change whose
  effect depends on the model (input shapes or layouts, calibration, conversion, unsupported
  operators) needs models that take the changed path. p50 should stay within run-to-run noise
  (about ±1% for yolov8n, ±2% for MobileNetV2) unless the change is meant to move it, and then
  the evidence says by how much and why. If you cannot run them, say so: the maintainer will.
- **A guard test.** A change that touches an invariant adds or updates a test that fails if the
  invariant breaks.
- **Golden fixtures unchanged.** They are never regenerated to make a test pass; changing one
  is a deliberate decision about the result's schema or behaviour, stated as `intended`.

## Licence

Contributions are made under the repository's Apache 2.0 licence: [LICENSE](LICENSE).
