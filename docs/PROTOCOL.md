# The measurement protocol

This document says how `calbredge run` measures a model: what is timed, the state the target is
in, what is recorded with each run, and when two runs compare. `docs/HOW_IT_WORKS.md`
introduces the concepts, `docs/RESULT_SCHEMA.md` defines every recorded field, and
`docs/SUPPORTED_PLATFORMS.md` gives the tier of each target, runtime and precision.

## What a run measures

A run measures the latency of one inference of one model, repeated: the wall-clock time from
handing an input tensor to the runtime until its outputs are back on the host. Each measured
iteration is timed on its own with `time.perf_counter` and kept.

The timer excludes loading the model, building or loading a compiled engine, warm-up, and
everything an application does around inference: decoding, preprocessing and postprocessing.
The input is a synthetic random tensor, generated once before warm-up and reused for every
iteration, so no data loading is timed. The result records this as `preprocessing: none`, with
the workload type `single_forward_pass`.

## The protocol file

A protocol is a YAML file that says how a run is conducted. `calbredge run --protocol <file>`
names one; without it the bundled `v1.0.yaml` is used.

| Field | Default | What it does |
|---|---|---|
| `protocol_version` | required | Recorded with the result. Every bundled protocol says `"1.0"` |
| `warmup_iterations` | 50 | Inferences run and discarded before measurement; 0 or more |
| `measurement_iterations` | 500 | Timed inferences; at least 1. The run has exactly this many samples |
| `batch_size` | 1 | Used for throughput only (see "Statistics and run status") |
| `input_shape` | required | The input tensor's shape, in the model's own axis order; every dimension positive |
| `power_mode` | `"default"` | Accepted and not acted on: a run records the power mode the target is in |
| `runtime_config` | `{}` | Accepted and not acted on |

A file with any other key is refused. The protocols bundled with the `calbredge-protocol`
package differ only in `input_shape`:

| File | `input_shape` |
|---|---|
| `v1.0.yaml` (the default), `protocol224.yaml` | 1, 3, 224, 224 |
| `protocol224_nhwc.yaml` | 1, 224, 224, 3 |
| `protocol320.yaml` | 1, 3, 320, 320 |
| `protocol320_nhwc.yaml` | 1, 320, 320, 3 |
| `yolo640.yaml` | 1, 3, 640, 640 |

Any other shape needs a protocol file of its own.

Every run records the protocol's digest, `protocol.protocol_digest`: a SHA-256 over the
canonical JSON of the validated protocol, defaults included. Comments, formatting and whether
a default is written out do not change it; any value does, including the two fields a run does
not act on.

**How `input_shape` is used.** TensorRT builds its engine for one shape: the model's static
dimensions, with each dynamic dimension taken from the protocol. ONNX Runtime feeds the same
shape: the model's static dimensions, with each dynamic dimension taken from the protocol, and
it refuses a protocol shape of another rank or one that differs from a static dimension. RKNN
feeds a tensor of the protocol's shape. The result's `protocol.input_shape` is derived from the `.onnx` for every runtime, with
a dynamic dimension bound from the protocol.

## The order of a run

1. **CPU affinity** is applied to the process (see "CPU affinity").
2. **The protocol** is loaded and validated.
3. **Identity** is collected: system metadata, the model file's SHA-256 and the `hardware_id`.
4. **The model is loaded.** TensorRT first refuses FP16 or INT8 on a GPU it reports has no
   fast support for that precision, then loads the cached engine whose build key matches, or
   builds one. RKNN checks the `.rknn` against its build record, loads it and runs one
   inference as a load-time check. ONNX Runtime creates its session.
5. **The input** is generated.
6. **Warm-up** runs `warmup_iterations` inferences. None of it is timed.
7. **The measured loop** runs `measurement_iterations` inferences, each timed.
8. **Telemetry and clock state** are read (see "Power" and "The performance state").
9. **Statistics** are computed from the samples, and the result and its run bundle are written.

## Warm-up and the timed loop

Warm-up and measurement use the same input and the same loaded model. What the timer encloses
differs by runtime only in what that runtime needs to finish the work:

| Runtime | Warm-up | Inside the timer, per measured iteration |
|---|---|---|
| TensorRT | The input is copied to the device once, the iterations are queued on the CUDA stream, and the stream is synchronized once at the end | The host-to-device copy of the input, the execution, the device-to-host copy of every output, and `stream.synchronize()` |
| ONNX Runtime | `session.run` for each iteration, then one CUDA synchronization when the CUDA provider is active | `session.run`, which returns the outputs as host arrays, then a CUDA synchronization when the CUDA provider is active |
| RKNN | `inference` for each iteration | One `inference` call, which is synchronous |

**The device is synchronized before the timer stops.** TensorRT's execution and copies are
asynchronous: without the synchronization the timer would stop when the work was queued, not
when it finished. Both copies are inside the timed region, so a sample is the time until the
outputs are usable on the host.

On ONNX Runtime the extra CUDA synchronization is `torch.cuda.synchronize`, and it runs only
when PyTorch is installed with CUDA. Without it the loop runs with no extra synchronization,
and the result does not say so.

## Statistics and run status

From the samples of the measured loop a run computes p50, p90 and p99 (`numpy.percentile`,
linear interpolation), the mean and the population standard deviation. Nothing is trimmed or
discarded. Throughput is `batch_size / (p50 / 1000)`, in inferences per second.

p50 is the number to compare runs on. p99 moves several times more between runs (see
"Run-to-run variance").

`run_status` is derived from the ratio of p99 to p50: above 3.0 the run is `unstable`, above
2.0 `inconclusive`, otherwise `stable`, with the ratio in `run_status_reason`. A run that is
not stable is still recorded; it is too noisy to compare.

## The performance state

The performance state is the power mode the target runs in, with its processor speeds held
fixed. `calbredge run` does not set it and does not refuse a target that is not in it: it
measures, and records the power mode and the clock state it found. `calbredge doctor` checks
the state and prints the fix, and `calbredge init` stops when the clocks are not pinned.

`target.clocks_pinned` is derived from the recorded clock state, by a rule per platform. It is
`null` when the state could not be read; pinning is never assumed. Every read the rule uses
works without root.

**Jetson.** `power_mode` is the mode `nvpmodel -q` reports, in canonical form (`maxn_super`).
`clocks_pinned` is true when every CPU policy has its minimum frequency equal to its maximum
and the GPU has its minimum equal to its maximum, which is how `jetson_clocks` leaves them; the
governor is not consulted. `examples/systemd/calbredge-jetson-clocks.service` runs
`jetson_clocks` at every boot. Changing the power mode unpins the clocks, and the power mode
is part of the TensorRT engine's build key, so another mode gets its own engine. The GPU node
is recognized on Orin; on other Jetson generations `clocks_pinned` is `null`. The memory clock
is recorded separately when it can be read, which needs root, and is not part of
`clocks_pinned`.

**RK3588.** The board has no named power modes, so `power_mode` is `not_applicable`.
`clocks_pinned` is true when every CPU policy's governor, the NPU's and the memory
controller's are `performance`, and every CPU policy's scaling maximum equals its hardware
maximum. The GPU does not run the model; its governor is recorded and not required.
`scripts/performance_mode.sh` sets that state, and
`examples/systemd/calbredge-rk3588-clocks.service` runs it at every boot.

**x86 Linux, and any other Linux host.** `power_mode` is `not_applicable`. The CPU governors
are recorded in the clock state, and `clocks_pinned` is `null`: no rule shows the speed held
fixed. Even with every governor at `performance` and turbo boost off, an x86 CPU can still
change speed with temperature and power limits. An NVIDIA GPU's clock state is not recorded
either. Such a run is not comparable with one whose speeds were locked. Every
x86 combination is results only.

The setup commands for each target are in `docs/QUICKSTART.md`, and the full rules in
`docs/RESULT_SCHEMA.md` §3.4.1.

## CPU affinity

On a CPU with fast and slow cores, which cores the process lands on changes the latency from
run to run: unpinned, latency on the RK3588 moves about 9% between runs. `--cpu-affinity` therefore
pins the process, and its default is `big`.

| Value | What is pinned | Recorded tier |
|---|---|---|
| `big` (default) | Every core at the highest capacity the kernel reports (`cpu_capacity`) | `big` |
| `little` | Every other core | `little` |
| a core list, such as `4,5,6,7` | Those cores | `custom` |
| `none` | Nothing | `none` |
| `big` or `little` where no capacity difference is found | Nothing | `homogeneous` |

The pin is applied with `sched_setaffinity` before the model is loaded, so the threads a
runtime starts while loading inherit it. On the RK3588, `big` is the four Cortex-A76 cores. A
Jetson Orin Nano's cores are all the same, and x86 hosts report no capacities, so both record
`homogeneous`; that includes an x86 CPU with two kinds of core.

The result records the request, the cores applied and the tier (`target.cpu_affinity`,
`target.cpu_affinity_tier`).

## What is recorded

Every run writes a result and a run bundle; `docs/RESULT_SCHEMA.md` is the contract. In outline:

- **What ran:** the model's SHA-256; the runtime label, as the adapter that ran reports it
  (`tensorrt-fp16`, `rknn-fp16`, `onnxruntime-cpu`, `onnxruntime-cuda`); the runtime's version;
  the precision; for a compiled engine, its hash and its build record (`engine_build`: the
  build key, the model it was built from, the TensorRT version or the RKNN conversion's input
  handling, the calibration digest); the git commit of the working directory.
- **Where:** `hardware_family` and `hardware_id`, the power mode, the clock state and
  `clocks_pinned`, the CPU affinity.
- **How:** the protocol's version, iterations, batch size and digest; the model's input shape
  and input and output signature.
- **What was measured:** p50, p90, p99, mean, standard deviation, the sample count, throughput
  and `run_status`; power, energy and temperature where the target reports them, with the
  instrument (`power_source`), the point measured (`power_domain`) and when it was read
  (`telemetry_measurement_boundary`).
- **The samples:** `samples.jsonl` in the bundle holds the latency of every measured
  iteration at full precision, so the statistics can be recomputed.

Three rules hold for every field. A value that could not be read is `null`, never 0 and never
a placeholder. A field that describes the code path, such as the runtime label or the
telemetry boundary, is set by the path that ran and never from user input. Nothing is
substituted silently: ONNX Runtime falling back from CUDA to the CPU shows in the runtime
label, a cached engine whose build record does not match is refused, FP16 or INT8 on a GPU
without fast support for it is refused rather than run with other kernels, and a `.rknn` with
no build record runs with a warning and records no `engine_build`.

## When two runs compare

Two p50 values are comparable when the runs measured the same work, the same way, in the same
state. Calbredge decides nothing here by itself: it records the facts, called compatibility
keys, and two results compare when every key is equal in both.

| Key | What a difference means |
|---|---|
| `protocol.protocol_digest`, `protocol.protocol_version` | A different procedure: iterations, batch size or input shape |
| `protocol.input_shape`, `protocol.io_signature` | A model with different inputs or outputs |
| `candidate.runtime`, `candidate.runtime_version` | A different runtime, or another version of it |
| `candidate.precision` | A different operating point (FP16 and INT8) |
| `candidate.engine_build.conversion_input` (RKNN) | The model's inputs were converted differently |
| `target.hardware_family` | A different kind of target |
| `target.power_mode` | A different power mode |
| `target.clocks_pinned` | Must be `true` in both |
| `target.cpu_affinity_tier` | Different cores; for `custom`, the applied cores must match too |

A `null` never equals another `null`: a key that was not recorded makes the runs
incomparable. Both runs should be `stable`.

Two fields are deliberately not keys. `candidate.model_hash` says whether the model file
changed, which is usually the change being measured. `target.hardware_id` identifies one
device by its hostname and hardware, so two identical boards differ in it; `hardware_family`
is the key.

To compare power or energy, `target.power_domain` and
`target.telemetry_measurement_boundary` must match as well.

`docs/QUICKSTART.md` shows the check on two results.

## Power

Power is measured where the target offers a way to, and each result says with what, at which
point and when.

| | Jetson | RK3588 | x86 |
|---|---|---|---|
| Instrument (`power_source`) | `tegrastats` | `external_meter`: a KM003C USB meter | none |
| Point measured (`power_domain`) | `module_rail`: the module's `VDD_IN` input, not the carrier board | `board_input`: the board's 5 V USB-C input, the whole board | n/a |
| When (`telemetry_measurement_boundary`) | `post_loop_snapshot`: 2 s after the measured loop ends | `windowed_bracket`: from the end of warm-up to the end of the last measured iteration | n/a |
| Sampling | Every 100 ms | Polled every 50 ms; the meter updates at about 10 Hz | n/a |
| `watts_average`, `watts_peak` | Mean and maximum of the samples | Time-weighted mean and maximum | `null` |
| `energy_joules_window` | `null` | Power integrated over the window | `null` |
| Temperature | Peak GPU temperature in the same 2 s (CPU if the GPU's is not reported); throttling is flagged above 85 °C | not recorded | `null` |

`joules_per_inference` is `watts_average` multiplied by the p50 latency, on both targets.

**Jetson.** The reading is taken after the timed loop, so it describes the module just after
the work, not during it, and `joules_per_inference` is an approximation. `calbredge run`
labels the power line as provisional and leaves energy per inference off the console; the
recorded values are as read. Latency is unaffected.

**RK3588.** The meter is optional: it needs the `power-meter` extra of `calbredge-protocol`
and a meter on the board's power input. Without either, the run completes and records no
power. The sampler runs in a background thread and never fails a run. A window that yields a
single sample records its energy as 0.

Compare power only between runs with the same `power_domain` and the same
`telemetry_measurement_boundary`: a module-rail figure read after the loop and a whole-board
figure sampled during it are different measurements.

## Run-to-run variance

How much p50 moves between runs of an unchanged model sets the smallest difference that means
anything. The figures measured so far:

| Model | Conditions | p50 between runs | p99 between runs |
|---|---|---|---|
| yolov8n | Jetson Orin Nano Super, TensorRT FP16, clocks pinned | about 0.3% | up to about 6% |
| MobileNetV2, about 1 ms | the same | under 2% (1.84% between two consecutive runs) | not stated |
| yolov8n | Orange Pi 5 Plus (RK3588), RKNN FP16, CPU affinity `big`, clocks pinned by the boot unit; 12 runs per model over two sessions, all stable | 54.66 to 55.84 ms, about 2% | not stated |
| MobileNetV2 | the same | 5.93 to 6.03 ms, about 1.7%, in two modes near 5.93 and 6.00 ms | not stated |

Small, fast models move more. These figures hold for the conditions named. They come from a
small number of runs on one board and are provisional until every published figure is
measured again on the final measurement code. They do not apply to x86, where the speeds are
not fixed.

To learn the movement on your own target, run the unchanged model three times in the state
you will compare in. A p50 difference inside that movement is not a change.

## Limits

- **Warm-up is a fixed count.** Nothing checks that the runtime reached a steady state before
  measurement starts.
- **The starting temperature is not controlled or recorded.** A target that has been working
  runs differently from a cool one.
- **One measurement at a time.** Two runs on one target at once skew each other; nothing
  prevents it.
- **x86 clocks.** Nothing locks an x86 CPU's speed, so x86 runs record `clocks_pinned` as
  `null` and compare only with care (see "The performance state").
- **Jetson memory clock.** A run with the CPU and GPU pinned and the memory clock free counts
  as pinned.
- **Jetson power is read after the loop** (see "Power").
- **TensorRT INT8 is calibrated on random noise:** 500 unseeded random batches of the model's
  input shape, whose recipe is part of the engine's build key. The engine therefore does not
  stand for a model calibrated on real data, and INT8 is results only.
- **ONNX Runtime and a model's other inputs.** The protocol's `input_shape` describes the
  first input. A second or later input is fed 1 for a dynamic dimension.
- **Protocol fields that are not acted on.** `power_mode` and `runtime_config` change nothing
  in a run but are part of the protocol's digest, so two protocols that differ only in them do
  not compare.
- **Batch size.** Every bundled protocol uses 1, and `batch_size` is not checked against the
  input shape.
