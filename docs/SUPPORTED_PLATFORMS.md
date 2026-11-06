# Supported platforms

A target is the hardware and runtime a model runs on in production. Every combination of
target, runtime and precision is in one of three tiers:

- **Supported:** the measurement is validated on this target: the timed path, the rule for
  when the clocks count as pinned, and how the hardware is identified.
- **Results only:** runs are measured and recorded (with their run bundle), but the part of the
  measurement named below is not validated yet, so compare such runs with care.
- **Planned:** not measured by this release.

A combination that isn't listed here isn't supported; `calbredge doctor` reports the tier of
each runtime installed on a host. The separately licensed `calbredge-gate` package gates
supported combinations only. How it works: `docs/HOW_IT_WORKS.md`. Getting started:
`docs/QUICKSTART.md`.

## The tiers

| Target | Runtime | Precision | Tier |
|---|---|---|---|
| NVIDIA Jetson | TensorRT | FP16 | Supported |
| NVIDIA Jetson | TensorRT | FP32 | Supported |
| NVIDIA Jetson | TensorRT | INT8 | Results only |
| NVIDIA Jetson | ONNX Runtime (CUDA) | any | Results only |
| x86_64 Linux | ONNX Runtime (CPU) | any | Results only |
| x86_64 Linux | ONNX Runtime (CUDA) | any | Results only |
| x86_64 Linux | TensorRT | any | Results only |
| Rockchip RK3588 | RKNN | FP16 | Supported |
| Rockchip RK3588 | RKNN | INT8 | Results only |
| Hailo-8 | HailoRT | INT8 | Planned |

Why the results-only rows aren't supported yet:

- **Jetson, TensorRT INT8:** INT8 engines are calibrated on random noise, not representative
  data, so their numbers don't stand for the model you'd ship. They are supported once
  calibration on real data exists.
- **Jetson, ONNX Runtime (CUDA):** this path hasn't been validated on a Jetson.
- **x86_64, ONNX Runtime (CPU):** there is no rule for when x86 clocks count as pinned, so
  results record `clocks_pinned` as null, and how the hardware is identified hasn't been
  validated.
- **x86_64, ONNX Runtime (CUDA) and TensorRT:** as for the CPU, and the NVIDIA GPU's clock
  state isn't recorded either.
- **Rockchip RK3588, RKNN INT8:** `calbredge run` measures it, but the check made when the
  model loads can refuse a correct INT8 model, and whether the quantized model's outputs still
  match the original's is not measured.

Planned targets:

- **Hailo-8 (HailoRT):** INT8 only, so it needs an output-fidelity check before it can be supported.

## Rockchip RK3588, RKNN FP16 (supported)

Validated on the Orange Pi 5 Plus. A model is converted to a `.rknn` on an x86 host with
`calbredge convert`, which writes a build record beside it; the board checks that record
before loading and refuses a `.rknn` that was not built from the `.onnx` beside it. The
result records how the model's inputs were converted (layout, normalization, input type), so
two runs of differently converted models are not mistaken for the same work.

"Clocks pinned" on this board means the CPU clusters, the NPU and the memory controller at
their performance setting, with the CPU's maximum speed not lowered. The GPU does not run the
model and is recorded only.

Calbredge does not judge whether an FP16 model's outputs are right: that is your own quality
evidence. The check it makes when the model loads is informational for FP16.

## NVIDIA Jetson (supported)

Validated on a Jetson Orin Nano Super with JetPack 6.2 (Python 3.10, TensorRT 10.3). Any power
mode can be used: choose the one your product deploys in; every run must use the same one. Setup:
`docs/QUICKSTART.md`, and `docs/JETSON_SETUP.md` for details.

**What is measured, for every run:**

| What | How |
|---|---|
| Latency | Every measured iteration is timed individually, after warm-up; the run records p50, p90, p99, mean and standard deviation, and whether the run was stable |
| Throughput | Inferences per second at the median: batch size divided by the p50 latency |
| Power | The module's `VDD_IN` rail, read with `tegrastats`: the Jetson module's input, not the whole carrier board. Average and peak watts |
| Energy per inference | Average power multiplied by the p50 latency |
| Temperature | Peak GPU temperature (CPU if the GPU's isn't reported), from `tegrastats` |
| Thermal throttling | Flagged when the peak temperature exceeds 85 °C |
| Performance state | The power mode, and whether the CPU and GPU speeds were locked; the memory speed too where it can be read (it needs root) |
| What ran | Hashes of the model and the TensorRT engine, the TensorRT version, the protocol and its digest |

**Known limits:**

- **Power is read after the timed loop, not during it.** The power reading is a short sample
  taken once the measured iterations have finished, so it describes the target just after the
  work rather than during it. Until power is sampled across the timed loop itself, treat power
  and energy figures as approximate; latency is unaffected.
- **Thermal throttling is inferred from temperature.** It is not read from the clock governor.
- **INT8 is results only** until calibration on real data exists (see above).
- **One measurement at a time.** Two measurements on one target at once skew each other.
