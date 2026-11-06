# Result Schema v2

This document is the contract for `result.json` v2, the measured-evidence record Calbredge writes
for every benchmark run. The Pydantic model and this document are maintained together; a change
to one without the other is a bug.

A compatibility key is a field that must be equal in two results for them to be compared; the
sections below say which fields are keys where they define them.

---

## 1. Scope

- **`result.json` v2 is measured evidence only.** It is written once and never rewritten. A
  judgement about a result is a separate record, and one result may be judged many times. A
  separately licensed package, `calbredge-gate`, adds release gating on top of these
  measurements; it is not part of this repository.
- **v2 extends the v1 `BenchmarkResult`; it does not replace it.** Every v1 field has a v2
  location (the "v1" column below), except `status`/`status_reason`, which `run_status`
  supersedes (§3.7). The upload format is still v1 `BenchmarkResult`, unchanged.
  v2 is a local artifact.
- **`schema_version` is separate from `protocol_version`.** `schema_version` versions the
  *shape of this record*. `protocol_version` versions the *measurement procedure*. Either can
  change without the other.
- **Schema evolution.** `schema_version` is `"MAJOR.MINOR"`, always a quoted string. The same
  rule applies to every Calbredge record that carries a `schema_version`:
  - A **minor** bump only ever adds **optional** fields. It never removes, renames or retypes a
    field, and never makes an optional field required.
  - A **major** bump is any other change. A reader rejects a major version it does not know.
  - A reader rejects a **newer minor** of its own major, with a message to upgrade Calbredge.
    It does not read the record and ignore the fields it doesn't know: those fields were part
    of what the newer writer recorded, and dropping them silently could change what the record
    means.
  - A reader accepts every older minor of its own major. Version 2.0 records must load in every
    future 2.x reader.
  - **Current version: 2.3.** 2.1 added the optional `candidate.engine_build` (§3.3.2); 2.2 the
    optional `protocol.protocol_digest` (§3.5); 2.3 the optional
    `candidate.engine_build.conversion_input` (§3.3.2).
- A v2 result lives in a run bundle, `<bundle dir>/<result_id>/` (default
  `~/.calbredge/runs/`). `result.json` is one file in that bundle; raw per-iteration samples live
  next to it in `samples.jsonl`. See §6.

## 2. Conventions

**Units go in field names.** `latency_p50_ms`, `temperature_peak_celsius`, `throughput_ips`.
Values are bare numbers; there is no `{value, unit}` wrapper.

**Full precision is stored.** "Display" below is how many decimal places tools show. It is
presentation only: comparisons always use the stored, unrounded value.

**Canonical string forms.** Every string field that is ever compared has exactly one canonical
form, and comparison is on that form only:

| Kind | Rule | Example |
|---|---|---|
| Enum-like identifiers | Lowercase, as recorded. Free-text sources are sanitized: lowercase, then every run of non-alphanumeric characters becomes `_`, with leading/trailing `_` stripped. | `maxn_super` (never `MAXN_SUPER`), `fp16`, `module_rail` |
| Runtime labels | Lowercase, hyphen-qualified, exactly as the adapter reports it | `tensorrt-fp16`, `onnxruntime-cpu`, `rknn-int8` |
| Version strings | The exact recorded string. No case folding, no numeric coercion (`"1.0"` ≠ `"1"`) | `"1.0"`, `"10.3.0"` |
| Digests | Lowercase hex SHA-256, 64 characters, no prefix | `9f86d0…` |
| Decision tokens | Uppercase (they are verdicts and statuses, not identifiers) | `NOT_EVALUATED`, `PASS` |

Human-facing output may display a friendlier form (for example `MAXN_SUPER`); stored data never
does.

**Null means "not recorded".** A null is never a wildcard and never equal to another null for
comparison purposes. A value that could not be determined is recorded as `null`, never as a
placeholder string such as `"unknown"`.

**Source kinds.** Each field lists where its value comes from:

| Source | Meaning |
|---|---|
| collected | Read from the host at run time (OS, sysfs, device tree, driver tools) |
| derived | Computed deterministically from other recorded data or from the artifact bytes |
| code-path-set | Set by the code path that actually executed. It is never defaulted and never taken from user input |
| declared | Supplied by the user via the protocol file, manifest or CLI |

**Required** means the field is always present and non-null in a v2 result produced by v2 code.
**Optional** fields are always present but may be `null`.

---

## 3. Fields

### 3.1 Top level

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `schema_version` | string | n/a | required | code-path-set: the version of the code that wrote the record, `"2.3"` today (§1) | new |
| `result_id` | string (UUID4) | n/a | required | derived at run time | `run_id` |
| `timestamp` | string (ISO 8601, UTC) | n/a | required | collected | `timestamp` |

### 3.2 `workload`

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `workload_type` | enum: `single_forward_pass` \| `iterative_action_policy` | n/a | required | code-path-set (the runner that executed) | new (v1 is implicitly `single_forward_pass`) |
| `work_unit` | enum: `inference` \| `decision` | n/a | required | code-path-set: `inference` for `single_forward_pass`, `decision` for `iterative_action_policy` | new |
| `workload_params` | object | per key | optional | declared | new |

`work_unit` is the thing one latency sample measures: one forward pass (`inference`), or one
complete action-chunk decision including every denoising step (`decision`).

`workload_params` is opaque to the shared schema; each workload contract defines its keys. For
`iterative_action_policy` they include `scheduler`, `denoising_steps`, `action_horizon`,
`control_rate_hz`, `replan_budget` and `fidelity_metric`.

### 3.3 `candidate`

What is being evaluated: the model, its compiled form, the code that produced it, and the
runtime it executes on.

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `model_hash` | string (digest) | n/a | required | derived: SHA-256 of the model file bytes | `model_hash` |
| `model_label` | string | n/a | optional | declared, stored as-is (display only, never compared) | `model_label` |
| `engine_hash` | string (digest) | n/a | optional | derived: SHA-256 of the compiled engine/plan file the runtime loaded. `null` for runtimes that execute the model file directly (ONNX Runtime) | new |
| `git_commit` | string (40-char lowercase hex) | n/a | optional | collected from the working directory Calbredge was invoked in; `null` outside a git repository | new |
| `git_dirty` | boolean | n/a | optional | collected alongside `git_commit`: `true` if a **tracked** file had uncommitted changes (modified or staged). Untracked files are ignored, so build output does not mark a run dirty. `null` outside a git repository | new |
| `runtime` | string (runtime label) | n/a | required | code-path-set: the executing adapter's own label | `runtime` |
| `runtime_family` | enum: `onnxruntime` \| `tensorrt` \| `rknn` \| `tflite` | n/a | required | derived: the part of `runtime` before the first `-` | new |
| `runtime_version` | string (version) | n/a | required | collected from the installed runtime package | `system_metadata.runtime_version` (v1 may hold `"unknown"`; v2 never does) |
| `driver_version` | object, e.g. `{"l4t": "36.4.7", "cuda": "12.6.11"}` | n/a | optional | collected; only the keys that could be read: `l4t` (from `/etc/nv_tegra_release`), `cuda` (from `/usr/local/cuda/version.json`), `nvidia_driver` (from `nvidia-smi` where it exists: discrete GPUs, and Jetson on JetPack 6). `null` when none could be read | new |
| `precision` | enum: `fp32` \| `fp16` \| `int8` | n/a | required | TensorRT and RKNN: code-path-set, the adapter's configured precision (the suffix of `runtime`). Runtimes whose label has no precision suffix (ONNX Runtime): derived from the model graph, see §3.3.1 | new (v1 `hardware_metrics.precision` exists but is never populated; v1 encodes precision only in the `runtime` suffix) |
| `engine_build` | object: `{build_key, built_from_model_hash, tensorrt_version, calibration_digest, conversion_input}` | n/a | optional (added in 2.1; `conversion_input` in 2.3) | code-path-set when a compiled engine or a converted model with a build record was loaded (TensorRT, RKNN); `null` for runtimes that execute the model file directly. `build_key` is the engine cache key (§3.3.2); `built_from_model_hash` must equal `model_hash`, or the result is invalid; `tensorrt_version` and `calibration_digest` are `null` where they don't apply | new |

#### 3.3.1 `precision` for runtimes without a precision suffix

The model's input dtype is **not** used: a quantized ONNX model normally keeps `float32` inputs
and quantizes inside the graph, so the input dtype would label an INT8 model `fp32`. Instead:

1. If the graph contains QDQ operators (`QuantizeLinear`, `DequantizeLinear`,
   `DynamicQuantizeLinear`), integer compute operators (`MatMulInteger`, `ConvInteger`) or any
   `QLinear*` operator, anywhere including subgraphs, `precision` is `int8`.
2. Otherwise, `precision` is the dtype of the **dominant initializer** type: the element type
   holding the largest total number of weight elements, counting initializers and `Constant`
   nodes (`float32` → `fp32`, `float16` → `fp16`, `int8`/`uint8` → `int8`). A graph with no
   weights, or whose dominant type is none of these, has no derivable precision (`null`), so it
   cannot produce a valid v2 result on a runtime without a precision suffix.

#### 3.3.2 Compiled engines and `engine_build`

A compiled engine is reused only when it was built from exactly the same inputs. Its cache key,
`build_key`, is the SHA-256 of the canonical JSON (RFC 8785) of:

| Key input | Why it matters |
|---|---|
| `model_hash` | A retrained model with the same inputs and outputs is a different engine |
| `precision`, `power_mode` | Different kernels and clocks |
| input profile (min/opt/max shape per input) | Engines are built for specific shapes |
| calibration digest | INT8 engines depend on their calibration recipe; `null` otherwise |
| compiler version (TensorRT) | Serialized engines are version-specific |
| builder configuration version | Bumped when the builder's flags or settings change |

The engine file name carries the first 16 hex characters of `build_key`
(`<model>.onnx.trt.<precision>.<power_mode>.<build_key[:16]>.engine`), with a build record
beside it (`.engine.json`) holding every input. Any change in the inputs produces a new file;
nothing is overwritten. On load, the key recomputed from the record must equal the record's
key and the key this run expects, or the engine is refused. After loading, the engine's own
input/output tensors are also checked against the model's `io_signature`; a mismatch makes the
result invalid.

A `.rknn` model is converted on an x86 host and copied to the board, which cannot recompute
its key. `calbredge convert` therefore writes a build record beside it (`<artifact>.json`):
the same kind of `build_key` over the source model hash,
precision, target platform, the model's inputs as declared and as converted, normalization,
graph edits, calibration digest, rknn-toolkit2 version and converter configuration version,
plus the SHA-256 of the `.rknn` file, which ties the record to that one file.

On the board the RKNN adapter checks the record before it loads anything: the record's inputs
must hash to its key, the `.rknn` must be the file it names, and it must have been built from
the `.onnx` beside it, for this run's precision and target. A model re-exported after
conversion is refused. The result's `engine_build` then carries the record's `build_key`, the
source model hash and the calibration digest, with `tensorrt_version` `null`, and
`conversion_input`: `feed_dtype` (the input type the converted model takes), `mean_values` and
`std_values` (the normalization folded into it), `inputs` (each model input's `source_layout`
and `converted_shape`, in the model's input order) and `graph_edits` (sorted). It is a
compatibility key for RKNN. `conversion_input` is absent for
engines built on the device (TensorRT). A `.rknn` without a record still runs, with a warning
and `engine_build` `null`; such a result does not show which model the `.rknn` was built from.

Engines are cached next to the model when that directory is writable, otherwise in
`~/.calbredge/engine-cache/`. Engine files named without a build key are never read.

### 3.4 `target`

The hardware and measurement setup the candidate ran on.

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `hardware_id` | string (digest) | n/a | required | derived, see §4 | `hardware_id` |
| `hardware_family` | string (canonical identifier) | n/a | required | collected, see §4 | new |
| `hardware_family_source` | enum: `device_tree` \| `derived` | n/a | required | code-path-set, see §4 | new |
| `power_mode` | string (canonical identifier) | n/a | required | collected. Jetson: the `nvpmodel -q` mode name, sanitized (e.g. `maxn_super`). Hosts with no power-mode mechanism: `not_applicable` | `hardware_metrics.power_mode` (v1 is `null` off Jetson) |
| `power_domain` | enum: `module_rail` \| `board_input` | n/a | optional | code-path-set: the point in the power-delivery chain the executing sampler measured. `null` when no power was measured | `hardware_metrics.power_domain` (Jetson rows recorded `null` until the run-bundle change; `module_rail` since) |
| `power_source` | enum: `tegrastats` \| `external_meter` | n/a | optional | code-path-set: the instrument that measured power. `null` when no power was measured | `hardware_metrics.power_source` |
| `telemetry_measurement_boundary` | enum: `post_loop_snapshot` \| `windowed_bracket` | n/a | optional | code-path-set: `windowed_bracket` when telemetry sampling brackets the measurement loop; `post_loop_snapshot` when it is read after the loop ends. `null` when no telemetry was captured | new |
| `clock_state` | object | n/a | optional | collected: platform-tagged governor/clock state (`source`: `rk3588`, `jetson`, `linux_cpufreq`), or `{"captured": false, "reason": ...}` when it could not be read | `clock_state` (v1 records it on RK3588 only; Jetson and x86 rows are `null`) |
| `clocks_pinned` | boolean | n/a | optional (`null` when clock state was not captured, or the platform has no pinning rule) | derived from `clock_state`, see §3.4.1 | new |
| `cpu_affinity` | object | n/a | optional | code-path-set from the `--cpu-affinity` request | `cpu_affinity` |
| `cpu_affinity_tier` | enum: `big` \| `little` \| `custom` \| `none` \| `homogeneous` | n/a | required | derived from `cpu_affinity`, see §3.4.2 | new |
| `system_metadata` | object | n/a | required | collected: every v1 key (`hostname`, `os_name`, `os_version`, `cpu_model`, `cpu_cores`, `ram_gb`, `gpu_model`, `python_version`, `runtime_name`, `runtime_version`) plus the new `device_tree_model`. Compared values are the promoted copies (`candidate.runtime_version`, `target.hardware_family`), never these | `system_metadata`, unchanged, plus `device_tree_model` |

#### 3.4.1 `clocks_pinned`

`true` when every clock domain that affects the measured work was held at a fixed frequency for
the run, `false` when any of them was free to scale. Unpinned clocks change latency materially
(for example after a reboot resets them), so this is a compatibility key. Per platform:

| `clock_state.source` | `clocks_pinned` is `true` when |
|---|---|
| `jetson` | Every CPU cpufreq policy has `scaling_min_freq == scaling_max_freq`, **and** the GPU devfreq node has `min_freq == max_freq`, as `jetson_clocks` sets them. Not governor-based: the CPU governor stays `schedutil` when pinned. Other devfreq nodes (video, JPEG, optical flow) are ignored. The GPU node is recognized on Orin (`17000000.gpu`) only; on other Jetson generations `clocks_pinned` is `null` until their node is verified |
| `rk3588` | Every CPU policy governor, the NPU governor and the DMC (memory) governor are `performance`, **and** every CPU policy has `scaling_max_freq == cpuinfo_max_freq`: a `performance` governor under a lowered maximum is not pinned. The rule is the CPU, the accelerator that runs the model, and memory. The GPU is idle during RKNN inference, so its governor and frequency are recorded in `clock_state` and not required. `clock_state` also records each CPU policy's current, minimum and maximum scaling frequency and hardware maximum (kHz), and the NPU, DMC and GPU current and maximum frequency (Hz, `null` where unreadable). If the governors read pinned and the CPU frequencies were not recorded, `clocks_pinned` is `null`, never `true` |
| `linux_cpufreq` (other Linux hosts) | Never: `clocks_pinned` is always `null`. The governors are recorded in `clock_state`, and nothing is judged from them: even at `performance` with turbo boost off, the speed still changes with temperature and power limits |

If `clock_state` is `null` or `{"captured": false}`, `clocks_pinned` is `null`: pinning is never
assumed.

**EMC (memory clock) on Jetson.** EMC pinning is recorded separately, in
`clock_state.emc = {"pinned": true | false | null, "source": "debugfs" | null}`, and is **not**
part of `clocks_pinned`. It is read from the BPMP debugfs clock node, which requires root on
stock JetPack, so an unprivileged run records `pinned: null` with the reason. There is no
unprivileged indirect reading: `tegrastats` omits the EMC frequency when not run as root, and
the `MAXN_SUPER` power mode declares no EMC cap to compare against. **Accepted gap:** a run with
CPU and GPU pinned but EMC free counts as `clocks_pinned: true`.

All clock collection runs unprivileged.

#### 3.4.2 `cpu_affinity_tier`

Derived from the recorded `cpu_affinity`:

| `cpu_affinity` record | `cpu_affinity_tier` |
|---|---|
| `tier: "big"` | `big` |
| `tier: "little"` | `little` |
| `tier: "custom"` (explicit core list) | `custom` |
| `requested: "none"`, `applied: null` | `none` |
| `applied: null`, `reason: "homogeneous CPU"` | `homogeneous` |

For `custom`, the tier alone does not identify the placement; two runs compare only when
`cpu_affinity.applied` is equal too.

`power_domain`, `power_source` and `telemetry_measurement_boundary` describe how the power and
thermal metrics in §3.7 were measured. They are always set by the code path that ran, so a
result can never claim a measurement boundary its sampler did not have.

### 3.5 `protocol`

How the measurement was conducted.

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `protocol_version` | string (version) | n/a | required | declared (protocol file) | `protocol_version` |
| `warmup_iterations` | integer ≥ 0 | iterations | required | declared | protocol file only (not in v1 result) |
| `measurement_iterations` | integer > 0 | iterations | required | declared | protocol file only (v1 result has `sample_count`) |
| `batch_size` | integer ≥ 1 | n/a | required | declared | protocol file only |
| `input_shape` | array of positive integers, model-native axis order | elements per axis | required | derived from the source `.onnx` model, for every runtime (see below) | `input_shape` (v1 may be `null`, e.g. RKNN) |
| `io_signature` | object: `{inputs: [{name, dtype, shape}], outputs: [{name, dtype, shape}]}` | n/a | required | derived from the source `.onnx` model, for every runtime | new |
| `preprocessing` | string (canonical identifier) | n/a | required | declared; `none` when inputs are synthetic and unprocessed | new |
| `input_fixture_digest` | string (digest) | n/a | optional | derived: SHA-256 of the input fixture's bytes. `null` when inputs are synthetic random tensors | new |
| `graph_kind` | enum: `fixed_compute` \| `data_dependent` | n/a | required | derived from the model graph: `data_dependent` when it has an operator whose amount of work depends on input values, else `fixed_compute` | new |
| `protocol_digest` | string (digest) | n/a | optional (added in 2.2) | code-path-set: SHA-256 over the RFC 8785 JSON of the protocol the run loaded, defaults included, so formatting and comments don't change it; absent from results written before 2.2 | new |

**Both are derived from the source `.onnx`**, which every runtime receives: ONNX Runtime
executes it, TensorRT builds its engine from it, and RKNN locates its converted model from it.
Static dimensions come from the graph. A dynamic dimension is bound from the protocol's declared
`input_shape` in `input_shape`, and recorded by its symbolic name in `io_signature`.

`input_shape` keeps the model's own axis order: NCHW for most vision models, NHWC for
TF/Keras/TFLite-origin exports. Consumers must treat it as an ordered tuple and never assume
which axis is channels.

**RKNN:** `.onnx` → `.rknn` conversion may change the input layout (for example to NHWC).
RKNN results record the source `.onnx` contract here, and the layout the conversion used in
`candidate.engine_build.conversion_input` (§3.3.2).

### 3.6 `capabilities`

What the run required and exercised, as checked by capability validation before execution.

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `artifact_format` | enum: `onnx` \| `engine` \| `rknn` \| `tflite` | n/a | required | derived from the artifact loaded | new |
| `telemetry` | array of enum: `power`, `thermal`, `memory` | n/a | required (may be empty) | code-path-set: the telemetry this run actually captured | new |

### 3.7 `metrics` (shared by every workload)

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `latency_p50_ms` | number | ms / 3 dp | required | derived from samples | `latency_p50_ms` |
| `latency_p90_ms` | number | ms / 3 dp | required | derived | `latency_p90_ms` |
| `latency_p99_ms` | number | ms / 3 dp | required | derived | `latency_p99_ms` |
| `latency_mean_ms` | number | ms / 3 dp | required | derived | `latency_mean_ms` |
| `latency_std_ms` | number | ms / 3 dp | required | derived | `latency_std_ms` |
| `sample_count` | integer | samples | required | derived | `sample_count` |
| `throughput_ips` | number | work units per second / 1 dp | required | derived: `batch_size / (latency_p50_ms / 1000)` | `throughput_ips` |
| `watts_average` | number | W / 3 dp | optional | collected | `hardware_metrics.watts_average` |
| `watts_peak` | number | W / 3 dp | optional | collected | `hardware_metrics.watts_peak` |
| `energy_joules_window` | number | J / 3 dp | optional | derived: time-integrated energy over a bracketed window; only from a `windowed_bracket` sampler | `hardware_metrics.energy_joules_window` |
| `joules_per_inference` | number | J per work unit / 4 dp | optional | derived: `watts_average × latency_p50_ms / 1000`. Approximate under `post_loop_snapshot`, where `watts_average` is read after the loop rather than during it | `hardware_metrics.joules_per_inference` |
| `temperature_start_celsius` | number | °C / 1 dp | optional | collected | `hardware_metrics.temperature_start_celsius` |
| `temperature_end_celsius` | number | °C / 1 dp | optional | collected | `hardware_metrics.temperature_end_celsius` |
| `temperature_peak_celsius` | number | °C / 1 dp | optional | collected | `hardware_metrics.temperature_peak_celsius` |
| `thermal_throttle_detected` | boolean | n/a | optional | derived from `temperature_peak_celsius` | `hardware_metrics.thermal_throttle_detected` |
| `gpu_memory_used_mb` | number | MB / 0 dp | optional | collected | `hardware_metrics.gpu_memory_used_mb` |
| `gpu_memory_total_mb` | number | MB / 0 dp | optional | collected | `hardware_metrics.gpu_memory_total_mb` |
| `system_memory_used_mb` | number | MB / 0 dp | optional | collected | `hardware_metrics.system_memory_used_mb` |
| `run_status` | enum: `stable` \| `inconclusive` \| `unstable` | n/a | required | derived from the p99/p50 ratio (> 3.0 `unstable`, > 2.0 `inconclusive`, else `stable`) | `run_status` |
| `run_status_reason` | string | n/a | optional | derived; set when `run_status` is not `stable` | `run_status_reason` |
| `samples_pointer` | string (path relative to the run bundle) | n/a | required | code-path-set: `samples.jsonl` | new |

The v1 fields `status` and `status_reason` (the older p99/p50 classification) are not carried
into v2; `run_status` supersedes them.

**Samples are never inlined.** `samples.jsonl` holds one JSON object per measured iteration, and
`samples_pointer` refers to it. Summary statistics in `result.json` are computed from exactly
those samples.

### 3.8 `workload_metrics`

An object whose keys are defined by the workload contract, not by the shared schema. The unit
convention (§2) applies to every key.

For `iterative_action_policy`, keys include `T_decision_ms` (percentile summary of the full
decision), `T_encode_ms`, `T_denoise_step_ms`, `N_denoise`, `unet_calls_per_decision`,
`stale_actions_p50`, `stale_actions_p99` and the deadline fields. `single_forward_pass` defines no workload metrics.

### 3.9 `quality`

| Field | Type | Unit / display | Req | Source | v1 |
|---|---|---|---|---|---|
| `quality_status` | enum: `NOT_EVALUATED` \| `OUTPUT_FIDELITY_ONLY` \| `CUSTOMER_VALIDATED` | n/a | required | declared; `NOT_EVALUATED` unless quality evidence is supplied | new |

`quality_status` makes "not validated" explicit rather than implicit:

- `NOT_EVALUATED`: no task-quality evidence exists for this candidate.
- `OUTPUT_FIDELITY_ONLY`: outputs were compared against a reference implementation (for example
  FP32), but task quality was not measured.
- `CUSTOMER_VALIDATED`: task-quality evidence was supplied by the model owner.

---

## 4. `hardware_id` and `hardware_family`

These are two different things and must not be confused.

### `hardware_id`: device identity, not a compatibility key

`hardware_id` is the SHA-256 of

```
hostname | cpu_model | cpu_cores | ram_gb | gpu_model
```

where `cpu_cores` counts every CPU the kernel knows about, including offline ones, so power-mode
core hotplugging does not change it. It is stable across reboots, power-mode changes, OS
upgrades and runtime upgrades.

**`hardware_id` depends on the hostname.** Consequences:

- **Two physically identical boards get different `hardware_id`s** if their hostnames differ.
- **Renaming a host creates a new `hardware_id`** for the same physical device.
- There is no serial number in the hash.

`hardware_id` therefore answers "which device record is this?", not "is this the same kind of
hardware?". Comparing runs uses `hardware_family` instead.

### `hardware_family`: the compatibility key

- **When the host has a device tree** (Jetson, RK3588 and most ARM boards),
  `hardware_family` is the device-tree model string (`/proc/device-tree/model`) in canonical
  form, and `hardware_family_source` is `device_tree`. The raw string is kept in
  `system_metadata.device_tree_model`.
- **When it does not** (typically x86), `hardware_family` is derived from
  `cpu_model`, `cpu_cores`, `ram_gb` and `gpu_model`, joined with `|` and put into canonical
  form, and `hardware_family_source` is `derived`.
- The device tree always wins when present.

**The device-tree model string names the whole board, carrier included.** The same compute
module on two different carrier boards (for example a vendor developer kit and a third-party
carrier) reports two different strings, so it is two different `hardware_family` values. This
fails safe: two carriers are never treated as the same target. Grouping
equivalent boards into one family is not supported.

The derived form alone is not enough on embedded boards: two different modules can report the
same CPU model, core count, RAM size and GPU model string yet perform differently (for example,
different memory bandwidth or GPU clocks). The device-tree string distinguishes them.

---

## 5. Relationship to the upload format

Uploads use v1 `BenchmarkResult`. A v2 result carries
every v1 field except the superseded `status`/`status_reason` (see the "v1" columns), so a v1
record can always be produced from a v2 result (v1's `status` is recomputed from the p99/p50
ratio, as v1 computes it today).
The reverse is not true: a v1 record lacks required v2 fields (`hardware_family`, `graph_kind`,
`work_unit`, `io_signature`, `clocks_pinned` and others), so a v1 record cannot be upgraded into a valid v2
result.

---

## 6. Run bundle

`calbredge run` writes one bundle per run:

```
<bundle dir>/<result_id>/
    result.json          the v2 result (this document)
    samples.jsonl        one line per measured iteration
    environment.json     software environment and command line
    logs/calbredge.log   the measurement pipeline's progress lines
```

**Location.** The default bundle dir is `~/.calbredge/runs/`. `--bundle-dir <dir>` overrides it
(in CI, for example `--bundle-dir ./runs`, then upload `./runs` as an artifact). `--no-bundle`
writes nothing. Bundle messages go to stderr: `run`'s stdout is unchanged, and a bundle that
cannot be written is a warning, never a failed run.

**Written atomically.** A bundle is assembled under a hidden temporary name and renamed into
place, so `<result_id>/` is either complete or absent. An existing bundle is never overwritten.

**`samples.jsonl`.** One JSON object per measured iteration, in measurement order, warmup
excluded:

```
{"iteration": 0, "latency_ms": 4.4613}
{"iteration": 1, "latency_ms": 4.4498}
```

`latency_ms` values are written at full precision, so the latency statistics in `result.json`
are exactly reproducible from this file. It has `protocol.measurement_iterations` lines.

**`environment.json`.** Calbredge and Python versions, platform (`uname`), installed package
versions, the command line, and a fixed allowlist of environment variables
(`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `LD_LIBRARY_PATH`, `PYTHONHASHSEED`, and names
starting `ORT_`, `TRT_`, `TENSORRT_`). Secrets never appear in a bundle: the full environment is
never written; credential-looking flag values (`--api-key`, any `*key*`, `*token*`, `*secret*`,
`*password*` flag), `name=value` credentials and URL user-info in the command line are replaced
with `<redacted>`; and every file is scrubbed of the resolved API key and of the values of
credential-named environment variables before it is written.

`calbredge_version` is the installed distribution's version. Bundles written before the
September 2026 packaging fix record `"0.1.0"` whatever version was installed; for those, read
`packages["calbredge"]` in the same file (or the result's `git_commit`), and never group results
by `calbredge_version`.

**`result.invalid.json`.** If the assembled v2 result fails validation (for example, a required
field could not be determined on this host), the bundle holds `result.invalid.json` instead of
`result.json`: `{"errors": [...], "result": {...}}`. The v1 upload and `run`'s exit code are
unaffected. Nothing that reads v2 results accepts a result that is not valid v2.

**Not in the bundle yet.** `manifest.lock.yaml` and `report.md`.
