# Calbredge Documentation

The reference for how measurements are set up, what the power figures mean, how to read
results, and what each command does. Setup and the workflow are in `docs/QUICKSTART.md`, the
concepts in `docs/HOW_IT_WORKS.md`, and result fields and run bundles in
`docs/RESULT_SCHEMA.md`.

## Measurement setup

### Jetson performance state

A Jetson measures in a declared performance state: the power mode (`nvpmodel`) and pinned clocks
(`jetson_clocks`). Both are recorded on every run (`power_mode`, `clocks_pinned`) and both are
compatibility keys: two runs compare only when both match.

`clocks_pinned` is true when every CPU policy has `scaling_min_freq == scaling_max_freq` and the
GPU devfreq node has `min_freq == max_freq`; the governor is not consulted, because it stays
`schedutil` under `jetson_clocks`. Uncaptured clock state is null, never false
(`docs/RESULT_SCHEMA.md` §3.4.1).

`jetson_clocks` does not survive a reboot. `examples/systemd/calbredge-jetson-clocks.service`
is a oneshot unit that runs `jetson_clocks` at boot after `nvpmodel.service`. Changing the power mode with
`sudo nvpmodel -m <mode>` unpins the clocks, and because the unit stays active after its first
run, `systemctl start` does nothing; re-pin with `sudo systemctl restart
calbredge-jetson-clocks.service`. `calbredge doctor` and `init` print that
restart command when the unit is installed and `sudo jetson_clocks` otherwise.

### RK3588 governors

On the RK3588, `clocks_pinned` is true when every CPU policy governor, the NPU governor
(`fdab0000.npu`) and the DMC (memory) governor read `performance`, and every CPU policy's
scaling maximum equals its hardware maximum. The GPU (`fb000000.gpu`) is idle while the NPU
runs a model, so its governor is recorded and not required. `scripts/performance_mode.sh` (run
with sudo) sets that state until the next reboot, and
`examples/systemd/calbredge-rk3588-clocks.service` runs it at every boot, after
`cpufrequtils.service`. The governors and frequencies are recorded in the run's clock state;
`docs/RESULT_SCHEMA.md` §3.4.1 has the rule.

### x86 Linux CPU governors

On an x86 Linux host, and any Linux host other than a Jetson or an RK3588, the CPU governors
are recorded and `clocks_pinned` is null. No rule shows the speed held fixed: even with every
governor at `performance` and turbo boost off, the speed still changes with temperature and
power limits. `calbredge doctor` shows the clocks as a warning (`not determined`), and
`calbredge init` goes ahead. Every x86 combination is results only
(`docs/SUPPORTED_PLATFORMS.md`).

### Input shape and layout

The input shape comes from, in order: an explicit shape passed to the adapter, the protocol's
`input_shape`, and the default protocol (`v1.0.yaml`, 1x3x224x224) when no `--protocol` is given.
ONNX Runtime and TensorRT then read the shape the model actually binds and record that
(`resolved_input_shape`). RKNN leaves it unset and records the protocol's shape: the RKNN
adapter does not read back the shape the converted model binds, and the RKNN runtime accepts
wrong-sized input without error, so a completed run cannot prove which shape was consumed. For
RKNN, the protocol's shape and layout must therefore match the model.

A wrong shape fails before the timed loop with a size mismatch, for example
`param input size(150528) < model input size(1228800)`. The numbers are flattened element counts:
150528 is 224x224x3 and 1228800 is 640x640x3, and a ratio that is the square of a resolution
ratio means the height and width are wrong, not the channels or batch. Check the model's
declared input with `calbredge inspect --model <onnx>` or `calbredge doctor --model <onnx>`.

The bundled protocols (`protocol/calbredge_protocol/protocols/`) all measure batch 1 with 50 warm-up and 500
measured iterations. `calbredge init` picks the one whose shape matches the model's input:

| File | `input_shape` | Layout |
|---|---|---|
| `v1.0.yaml` (the default) | 1, 3, 224, 224 | NCHW |
| `protocol224.yaml` | 1, 3, 224, 224 | NCHW |
| `protocol224_nhwc.yaml` | 1, 224, 224, 3 | NHWC |
| `protocol320.yaml` | 1, 3, 320, 320 | NCHW |
| `protocol320_nhwc.yaml` | 1, 320, 320, 3 | NHWC |
| `yolo640.yaml` | 1, 3, 640, 640 | NCHW |

On the RK3588 the YOLO family and RT-DETR use `yolo640.yaml`, MobileNetV2 `protocol224.yaml`,
EfficientNet-B0 and the CXR MobileNetV2 `protocol224_nhwc.yaml`, and EfficientDet-Lite0
`protocol320_nhwc.yaml`. Any other shape needs a protocol file of its own; two runs compare only
when their protocols' digests match (`protocol_digest`).

## Power

### Jetson (tegrastats)

Jetson power is read with `tegrastats` at the module's input rail (`power_source=tegrastats`,
`power_domain=module_rail`): the compute module, not the carrier board. It is sampled every
100 ms for 2 s after the timed loop ends, not during it
(`telemetry_measurement_boundary=post_loop_snapshot`), so it reflects the state just after the
workload, and `joules_per_inference` (that reading times the p50) is an approximation. Until
power is sampled during the loop, `calbredge run` labels the power line
"provisional: read after the timed loop, not during it" and leaves J/inference off the console;
the recorded values are unaffected.

### RK3588 (external meter)

RK3588 power comes from a ChargerLAB KM003C meter (or a clone with the same USB interface) inline
on the board's 5V USB-C input (`power_source=external_meter`, `power_domain=board_input`): whole
board draw, including CPU, memory, NPU, peripherals and idle, so not comparable with Jetson
module figures. `PowerSampler` (`protocol/calbredge_protocol/power_meter.py`) brackets the measured iterations
(`telemetry_measurement_boundary=windowed_bracket`). Only one process may hold the meter.

The meter's reading updates at about 10 Hz, so a run's average rests on about ten samples per
second of measured loop. About 30 samples (MobileNetV2's roughly 3 s window) is the practical
floor; faster models, or INT8 runs at about half the latency, need more measured iterations for
a stable average. Absolute board power also carries a per-session offset of about 0.2 W (ambient
temperature, meter seating), uniform across models: trust relative power within a session over
absolute watts across sessions, report at most one decimal place of watts, and do not read
sub-0.2 W differences between sessions as real.

Without a meter, or without `pyusb` (`calbredge-protocol[power-meter]`), `hardware_metrics` is
null and the run completes normally.

### Comparing power

Compare power and energy only between runs with the same `power_domain` and the same
`telemetry_measurement_boundary`: a module-rail figure read after the loop and a whole-board
figure sampled during it measure different things. Both fields are recorded with every result.

## Interpreting results

Every measured iteration is timed and kept; p50, p90, p99, mean and standard deviation are
computed from all of them.

**p99 is a warning.** On a Jetson Orin Nano Super (yolov8n, TensorRT FP16, clocks pinned), p99
varies by up to about 6% between runs while p50 stays within about 0.3%. Read a change in p99 as
a hint; compare runs on p50.

**Small, fast models move more.** On the same Jetson, MobileNetV2 at about 1 ms moved 1.84% in
p50 between two consecutive runs. `docs/QUICKSTART.md` therefore gives "under 2%" as the
run-to-run movement of p50; learn your own target's by running the unchanged model three times.

**Compare power on multi-run means.** On the same setup, single-run Jetson power varies by about
1% (n=3), so a 1% difference between two single runs is noise. Compare power and energy on the
mean of at least five runs.

**Run status.** `run_status` is `unstable` when p99 is more than three times the p50,
`inconclusive` when more than twice, else `stable`. A run that is not stable is too noisy to
compare: check nothing else is using the target and measure again.

## CLI reference

`calbredge --help` lists six commands. Installed packages can add commands through the
`calbredge.commands` entry-point group. A command that belongs to the separately licensed
`calbredge-gate` package is refused with exit 2 and a message naming that package when it is
not installed. Unless stated otherwise, a command exits 0 on success, 1 on an error and 2 on a
usage error.

**`calbredge run`**: benchmark a model and save, print or upload the result.
`--model` (required; the `.onnx` for every runtime, see Known limitations), `--runtime`
(`onnxruntime`, `tensorrt[-fp32|-fp16|-int8]`, `rknn[-fp16|-int8]`, `tflite`, `mock`; detected
from the file extension if omitted), `--protocol` (default `v1.0.yaml`), `--model-label`,
`--cpu-affinity` (`big` by default: the highest-capacity cluster on big.LITTLE, a no-op on
homogeneous CPUs; `little`, `none`, or core ids; applied before the model loads),
`--bundle-dir` / `--no-bundle` (the v2 run bundle, default `~/.calbredge/runs/`,
`docs/RESULT_SCHEMA.md` §6), `--upload <full /api/benchmarks URL>`, `--api-key` (or
`CALBREDGE_API_KEY`), `--offline` (no network calls; overrides `--upload`), `--dry-run` (no
upload), `--debug`. `--api-key`, `--upload`, `--model-label` and `--runtime` fall back to
`~/.calbredge/config.yaml` when not given. The recorded runtime is the adapter's `runtime_label`
for TensorRT and RKNN, so bare `tensorrt` records `tensorrt-fp32` and bare `rknn` records
`rknn-fp16`; ONNX Runtime records `onnxruntime-cuda` or `onnxruntime-cpu` from the active
provider. A suffixed `--runtime` that disagrees with the adapter's label is an error. Results
are saved under `~/.calbredge/results/` whenever there is no upload or the upload fails. A
bundle that cannot be written is a warning, never a failure. Exit 1 on any load, configuration
or benchmark error.

**`calbredge init`**: make the current directory a project for one model, then measure it
once. `--model <onnx>` or `--example` (MobileNetV2 1.0, checked against a pinned SHA-256),
`--protocol`, `--runtime` (default `tensorrt-fp16` on a Jetson, `rknn-fp16` on an RK3588, else
`onnxruntime`), `--platform jetson|rk3588` (detected on a Jetson or an RK3588), `--force`. The
directory must be inside a git repository, its path may use only letters, digits, `.`, `_`, `-`
and `/`, and the model must be inside it (at any depth); a project can neither be inside
another project nor contain one. It writes `.calbredge/project.yaml` and `protocol.yaml` (the
bundled protocol matching the model's single input, or `--protocol`) here and extends
`.gitignore`. It refuses, writing nothing, when the clocks aren't pinned (exit 7), when no
bundled protocol fits, or, without `--force`, when the project exists or is nested. For an
RKNN runtime the model's `.rknn` and its build record must already be beside it (the board
cannot convert); `--example` is refused on an RK3588. Installed packages can add files and
options to `init` through the `calbredge.init_hooks` entry-point group.

**`calbredge doctor`**: check the host is ready to measure: clocks, power mode, hardware
family, TensorRT, pycuda, numpy, onnx and protobuf, power telemetry, the support tier of each
installed runtime, writable `~/.calbredge` directories, the config file (values never printed)
and package versions; on a Jetson also JetPack; on an RK3588 also `rknn-toolkit-lite2`,
`librknnrt.so` and its version, and access to the NPU. `--model` adds the model's inputs and
whether an engine is cached; `--json` prints one document. Read-only, never runs sudo. Exit 0
when ready (warnings allowed), 7 when a check fails.

**`calbredge inspect`**: print, as one JSON document, the v2 candidate, target and protocol
fields this host and model would record. `--model` (required), `--runtime`, `--protocol`,
`--cpu-affinity`. Read-only: it never loads the model into a runtime or reads the config file.
Fields only a run can set are null and listed under `inspect.not_collected`.

**`calbredge convert`**: convert a Keras (`.h5`) or PyTorch (`.pt`, `.pth`) model to ONNX, or an
ONNX model to RKNN (`--framework rknn`, x86 only). `--model`, `--output`, `--framework`,
`--input-shape` (PyTorch), `--precision` and `--calibration-dataset` (RKNN; the dataset is
required for INT8), `--target` (default `rk3588`), `--skip-validate-conversion`, `--debug`.

**`calbredge config`**: write `--api-key` and `--upload-url` to `~/.calbredge/config.yaml`.

## Known limitations

- **x86 clocks are never pinned.** x86 runs record `clocks_pinned` as null ("x86 Linux CPU
  governors" above). Do not compare such a run with one whose speeds were locked.
- **The RKNN canary falls back to noise for NCHW inputs.** The load-time correctness canary uses a
  real image (`canary_image_path`, or the first image in a `dataset.txt` beside the model) only
  for 3-channel NHWC shapes. For an NCHW shape it prints "is not (N, H, W, 3) ... Falling back to
  noise" and checks fixed-seed noise instead, which cannot catch some quantization failures. So
  NHWC models (EfficientNet-B0, CXR MobileNetV2, EfficientDet-Lite0) get the real-image check and
  NCHW models (the YOLO family, MobileNetV2) the weaker one.
- **The `.rknn` artifact is found by name.** RKNN loads `<model>.onnx.rknn.<precision>.<power_mode>`
  beside the model, matched by file name. Its build record ties it to the `.onnx`, and a `.rknn`
  built from another model is refused; a `.rknn` with no record still runs, with a warning.
  TensorRT engines are content-addressed (`docs/RESULT_SCHEMA.md` §3.3.2).
- **`--model` help advertises formats that are refused.** `calbredge run --help` lists `.engine`
  and `.rknn` as accepted, and extension detection maps them to `tensorrt` and `rknn`, but both
  adapters accept only the `.onnx` and raise `ValueError` for anything else. Pass the `.onnx`.
