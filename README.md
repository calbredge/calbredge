# Calbredge

Calbredge measures the performance of AI models on the edge hardware and runtime they actually
ship on, and makes runs comparable from one release to the next. It aims to replace hand-rolled
timing scripts, vendor spec sheets measured under ideal conditions, and spreadsheets used to
track deployments. It records latency (p50, p90, p99) and throughput on every supported target,
power and temperature on the Jetson, and power on the RK3588 with an optional external USB meter.

## How it works

### Protocol

Calbredge measures every run under a fixed protocol, so runs compare with each other and across
silicon. The default protocol runs 50 warm-up iterations, then 500 measured iterations at batch
size 1. Warm-up is never timed, and every measured iteration is timed individually. A run is
classified stable, inconclusive or unstable from its p99/p50 ratio (inconclusive above 2.0,
unstable above 3.0), so you know when a result is too noisy to use.

Full methodology: [docs/PROTOCOL.md](docs/PROTOCOL.md).

### Supported targets

Jetson with TensorRT (FP16, FP32) and RK3588 with RKNN (FP16) are currently supported. Other
combinations (TensorRT INT8, RKNN INT8, ONNX Runtime, and any runtime on an x86 host) are results
only: their runs are measured and recorded, but part of the measurement is not validated yet, so
compare them with care. The reasons are given in
[docs/SUPPORTED_PLATFORMS.md](docs/SUPPORTED_PLATFORMS.md).

## Quickstart

Clone the repository on each machine you use:

```bash
git clone https://github.com/calbredge/calbredge.git && cd calbredge
```

### Jetson (JetPack 6.2)

With the power mode chosen ([QUICKSTART step 1](docs/QUICKSTART.md#1-set-the-power-mode)):

```bash
python3 -m venv --system-site-packages ~/calbredge-venv
source ~/calbredge-venv/bin/activate
PATH=/usr/local/cuda/bin:$PATH pip install -c constraints/jetson-jp6.2-py310.txt ./protocol "./agent[jetson]"
sudo jetson_clocks          # lock the processor speeds until the next reboot
calbredge doctor
calbredge init --model models/<your model>.onnx
calbredge run --model models/<your model>.onnx --runtime tensorrt-fp16 --protocol .calbredge/protocol.yaml --offline
```

### RK3588

On an x86 host:

```bash
python3 -m venv ~/calbredge-venv
source ~/calbredge-venv/bin/activate
pip install -c constraints/rknn-convert-x86_64-py312.txt --extra-index-url https://download.pytorch.org/whl/cpu rknn-toolkit2 ./protocol "./agent[rknn-convert]"
calbredge convert --framework rknn --model <your model>.onnx --precision fp16 --output <your model>.onnx.rknn.fp16.default
# copy the .onnx, the .rknn and its build record to the board, side by side
```

On the board:

```bash
python3 -m venv ~/calbredge-venv
source ~/calbredge-venv/bin/activate
pip install -c constraints/rk3588-jammy-py310.txt ./protocol "./agent[rk3588]"
sudo scripts/performance_mode.sh          # lock the processor speeds until the next reboot
calbredge doctor
calbredge init --model models/<your model>.onnx
calbredge run --model models/<your model>.onnx --runtime rknn-fp16 --protocol .calbredge/protocol.yaml --offline
```

The full walkthrough, including comparing results: [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Commands

| Command | What it does |
|---|---|
| `calbredge run` | Measure a model and save the result and its run bundle |
| `calbredge init` | Make a directory a project for one model, then measure it once |
| `calbredge doctor` | Check the target is ready to measure, with a fix for anything that fails |
| `calbredge inspect` | Print what this host and model would record, without running |
| `calbredge convert` | Convert a model to ONNX, or an ONNX model to RKNN (on an x86 host) |
| `calbredge config` | Save an upload URL and API key, to upload results (optional) |

Every option: `calbredge <command> --help`.

## Status

Early and changing. Interfaces may change between releases. Issues are welcome; there is no
support commitment yet.

## How this was built

Much of Calbredge's code, tests and documentation was written with AI coding tools, directed
and reviewed by the founder, who makes every design and measurement decision. Changes that can
affect a measurement are checked by guard tests and, where they reach a target, by
before-and-after runs on the hardware.

## History

Calbredge began as a private project named EdgeBench in March 2026. This repository starts
from a snapshot of that work, so its history begins with its first commit.

## Licence

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
