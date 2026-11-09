# Quickstart

Calbredge measures how the AI models in robots, drones, cameras and other physical systems
perform on the hardware they will run on: inference latency under a fixed protocol, recorded
with everything needed to tell whether two runs compare. Every run records the model, the
runtime and its version, the protocol, the power mode and whether the processor speeds were
locked.

The hardware a model ships to is its target. Each walkthrough below takes one target from
setup to its first run, in the same four parts. The first two are setup, once per target and
once per model; the last two are what you repeat:

- **Set up your target**: put it in a declared performance state (the power mode your product
  deploys in, with the processor speeds locked) and install Calbredge on it.
- **Init a project** for your model: Calbredge writes its files and measures the model once.
- **Run** a measurement whenever the model, the runtime or the target changes.
- **Read and compare** the results.

What each of these terms means: `docs/HOW_IT_WORKS.md`.

## Contents

- [What's supported today](#whats-supported-today)
- [NVIDIA Jetson](#nvidia-jetson)
- [Rockchip RK3588, RKNN](#rockchip-rk3588-rknn)
- [x86_64 Linux, ONNX Runtime](#x86_64-linux-onnx-runtime)
- [Reading and comparing results](#reading-and-comparing-results)
- [More](#more)

## What's supported today

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

*Supported* means the measurement is validated on that target; *results only* runs are
measured and recorded, but part of the measurement is not validated yet; *planned* targets
aren't measured by this release. Jetson support is validated on JetPack 6.2 (Python 3.10).
Details: `docs/SUPPORTED_PLATFORMS.md`.

**You need:** a target from the table above, sudo on it, a clone of this repository on it, and a git
repository for the project (a new, empty one is fine for the example).

**Where the model lives is up to you.** The project's files name the model by path; the model
can be committed with them, stored with Git LFS, or kept out of git. Calbredge identifies the
model by its SHA-256, not by its location or by git.

**The performance state.** Two runs compare fairly only when the target was in the same
performance state for both:

- **The same power mode.** Choose the mode your product deploys in, and use it for every run
  you mean to compare. Calbredge records the mode with every run and never changes it.
- **Locked processor speeds.** Left alone, the processors speed up and slow down with load and
  temperature, and timings move with them. Locking the speeds holds them at the highest the
  chosen mode allows, so they can't change during a measurement. Results are then that mode's
  steady best case, not what a product whose speeds vary would see at every moment.

## NVIDIA Jetson

TensorRT, on JetPack 6.2. More on a Jetson (engines, the optional ONNX Runtime GPU build,
common errors): `docs/JETSON_SETUP.md`.

### 1. Set the power mode

A Jetson has modes such as 7W, 15W, 25W and MAXN_SUPER; the examples here use MAXN_SUPER. Mode
numbers differ between Jetson models, so read the mode's number from the target's own table.
Set `NAME` to the mode your product deploys in:

```bash
grep 'POWER_MODEL ID=' /etc/nvpmodel.conf
NAME=MAXN_SUPER
MODE=$(sed -n "s/.*POWER_MODEL ID=\([0-9]*\) NAME=$NAME\b.*/\1/p" /etc/nvpmodel.conf)
sudo nvpmodel -m "$MODE"
sudo nvpmodel -q
```

**Success:** `NV Power Mode:` followed by the mode you chose. If `nvpmodel` asks to reboot, say
yes (modes that take CPU cores offline, such as 7W, need one).

### 2. Install

Run these in the clone. The venv must see the system's packages, because TensorRT comes from
JetPack and can't be installed with pip. The constraints file holds the versions validated on
JetPack 6.2; pycuda compiles from source, which takes a few minutes and needs `nvcc` on `PATH`:

```bash
sudo apt-get install -y python3-venv python3-dev
python3 -m venv --system-site-packages ~/calbredge-venv
source ~/calbredge-venv/bin/activate
pip install --upgrade pip      # Ubuntu 22.04's stock pip can't install from a clone
PATH=/usr/local/cuda/bin:$PATH pip install -c constraints/jetson-jp6.2-py310.txt ./protocol "./agent[jetson]"
```

Activate the venv in every new shell (`source ~/calbredge-venv/bin/activate`).

### 3. Lock the processor speeds

NVIDIA's `jetson_clocks` locks the CPU, GPU and memory at the highest speed the chosen power
mode allows. This boot unit runs it now and at every boot:

```bash
sudo cp examples/systemd/calbredge-jetson-clocks.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now calbredge-jetson-clocks.service
```

Until the next reboot only, `sudo jetson_clocks` does the same:

```bash
sudo jetson_clocks
```

**Changing the power mode later.** Any `sudo nvpmodel -m` unlocks the speeds. After changing
the mode, lock them again:

```bash
sudo systemctl restart calbredge-jetson-clocks.service
```

### 4. Check the target

```bash
calbredge doctor
```

It checks what a measurement depends on (the clocks, the power mode, the runtime and its
versions) and prints a fix for anything that fails. It changes nothing and never runs sudo.

**Success:** it ends with `Ready.`, and `calbredge --version` prints the version you installed.

### 5. Init a project

Run `calbredge init` in the directory that is to be the project: `.calbredge/` is written
there, and the model must be inside it, in that directory or a folder below. The directory must
be inside a git repository.

```bash
calbredge init --model models/<your model>.onnx
```

or, to try the example model in a new repository:

```bash
mkdir ~/qs && cd ~/qs && git init -q
calbredge init --example
```

`--example` downloads MobileNetV2 1.0, an ImageNet classifier (224x224), from the
[ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet),
licensed Apache 2.0, and checks it against a pinned SHA-256 (`c0c3f76d…`); Calbredge doesn't
ship it.

`init` writes these files, then measures the model once (the discovery run; on a Jetson the
first measurement also builds the model's TensorRT engine, which takes a minute or so):

| File | What it is |
|---|---|
| `.calbredge/project.yaml` | The project: which model and protocol, and the target's runtime |
| `.calbredge/protocol.yaml` | The measurement protocol matching your model's input shape: warm-up and measured iterations, input shape |
| `.gitignore` | In the project's directory, created or extended: keeps the files runs leave behind (engines, caches) out of git |

**Success:** the files it wrote, `✓ OK discovery run …: stable, p50 … ms`, and the command that
measures the project again. If the run is unstable, `init` says so: check nothing else is using
the target and measure again. A model whose input shape matches no bundled protocol needs
`--protocol` with one you write (`calbredge init --help`). If the processor speeds aren't
locked, `init` stops before writing anything (exit 7) and says how to lock them.

Commit the project's files, so a change to how the model is measured is reviewed like any other
change.

### 6. Run

From the project's directory, with the runtime named in `.calbredge/project.yaml`:

```bash
calbredge run --model models/<your model>.onnx --runtime tensorrt-fp16 --protocol .calbredge/protocol.yaml --offline
```

`--offline` guarantees no network calls: the result stays on the target. The run warms the
model up, times every measured iteration individually, prints the result as JSON, and ends with
a summary. On a Jetson a `Hardware Metrics` block follows the summary, with power and
temperature: both are read just after the timed loop, so they are provisional until they are
sampled during it. What the summary says:
[Reading and comparing results](#reading-and-comparing-results).

## Rockchip RK3588, RKNN

The board runs a model as a `.rknn`, and can't convert one: the model is converted on an x86
host, then measured on the board. Bringing the board up (the OS image, the NPU driver, the
runtime library) is in `docs/ORANGE_PI_SETUP.md`.

### 1. Convert the model, on an x86 host

In a clone on the x86 host, install the converter with its constraints file, which holds the
versions tested with Python 3.12.3: RKNN-Toolkit2 2.3.2, `onnx` at exactly 1.18.0 (newer
releases break the toolkit's `load_onnx`), `setuptools` below 81 (the toolkit imports
`pkg_resources`, which 81 removed) and a `protobuf` the toolkit converts with as it is. The
extra index is PyTorch's CPU-only build: the toolkit requires `torch`, conversion uses no GPU,
and without the index pip installs the CUDA build: about 2.8 GB of downloads and 5.6 GB on disk,
against about 400 MB of downloads and 1.6 GB on disk with the CPU build.

```bash
python3 -m venv ~/calbredge-venv
source ~/calbredge-venv/bin/activate
pip install --upgrade pip      # older pip can't install from a clone
pip install -c constraints/rknn-convert-x86_64-py312.txt --extra-index-url https://download.pytorch.org/whl/cpu rknn-toolkit2 ./protocol "./agent[rknn-convert]"
```

Then convert. `calbredge convert` builds the `.rknn` under the name the board looks for, and
writes its build record beside it:

```bash
calbredge convert --framework rknn --model <your model>.onnx --precision fp16 --output <your model>.onnx.rknn.fp16.default
```

**Success:** `<your model>.onnx.rknn.fp16.default` and `<your model>.onnx.rknn.fp16.default.json`.
The record names the source `.onnx` by hash, the `.rknn` by hash, the toolkit version, the
precision and target, the model's inputs as declared and as converted, and the calibration
digest for an INT8 build.

Copy the `.onnx`, the `.rknn` and the record to the board together, the `.rknn` and the record
beside the `.onnx`: the record describes one `.rknn` built from one `.onnx`, and converting
again replaces both files. Two conversions of the same model give the same build key but not
the same bytes, so a record never fits a `.rknn` from another conversion.

### 2. Install on the board

With the board running, install in the clone:

```bash
sudo apt-get install -y python3-venv
python3 -m venv ~/calbredge-venv
source ~/calbredge-venv/bin/activate
pip install --upgrade pip      # Ubuntu 22.04's stock pip can't install from a clone
pip install -c constraints/rk3588-jammy-py310.txt ./protocol "./agent[rk3588]"
```

Activate the venv in every new shell (`source ~/calbredge-venv/bin/activate`).

### 3. Lock the processor speeds

`scripts/performance_mode.sh` sets the CPU, NPU and memory governors to `performance`; the boot
unit runs it now and at every boot:

```bash
sudo install -m 755 scripts/performance_mode.sh /usr/local/sbin/calbredge-performance-mode
sudo cp examples/systemd/calbredge-rk3588-clocks.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now calbredge-rk3588-clocks.service
```

Until the next reboot only, `sudo scripts/performance_mode.sh` does the same:

```bash
sudo scripts/performance_mode.sh
```

### 4. Check the board

```bash
calbredge doctor
```

It checks what a measurement depends on (the clocks, the power mode, the runtime and its
versions) and prints a fix for anything that fails. It changes nothing and never runs sudo.

**Success:** it ends with `Ready.`, and `calbredge --version` prints the version you installed.

### 5. Init a project

Run `calbredge init` in the directory that is to be the project: `.calbredge/` is written
there, and the model must be inside it, in that directory or a folder below. The directory must
be inside a git repository. The `.rknn` and its build record must be beside the `.onnx` before
`init`; `--example` is refused on an RK3588, which needs a converted model.

```bash
calbredge init --model models/<your model>.onnx
```

`init` writes `.calbredge/project.yaml`, `.calbredge/protocol.yaml` and the project's
`.gitignore` (what each is: [the Jetson's step 5](#5-init-a-project)), then measures the model
once (the discovery run).

**Success:** the files it wrote, `✓ OK discovery run …: stable, p50 … ms`, and the command that
measures the project again. If the run is unstable, `init` says so: check nothing else is using
the board and measure again. A model whose input shape matches no bundled protocol needs
`--protocol` with one you write (`calbredge init --help`). If the processor speeds aren't
locked, `init` stops before writing anything (exit 7) and says how to lock them.

Commit the project's files, so a change to how the model is measured is reviewed like any other
change. Commit the `.rknn` and its record with the `.onnx`: the board can't convert, and the
record names this exact `.onnx`, so the three travel together.

### 6. Run

From the project's directory, with the runtime named in `.calbredge/project.yaml`:

```bash
calbredge run --model models/<your model>.onnx --runtime rknn-fp16 --protocol .calbredge/protocol.yaml --offline
```

`--model` names the `.onnx`. The run checks the `.rknn` beside it against its record and the
`.onnx` before loading it, and refuses a mismatch with the check that failed. A `.rknn` with no
record runs with a warning, and its result does not show which model it was built from.

`--offline` guarantees no network calls: the result stays on the board. The run warms the
model up, times every measured iteration individually, prints the result as JSON, and ends with
a summary. What the summary says:
[Reading and comparing results](#reading-and-comparing-results).

## x86_64 Linux, ONNX Runtime

Every x86 combination is results only, and the processor speeds can't be locked here the way
they are on a Jetson or an RK3588: even with every CPU governor at `performance` and turbo
boost off, the speed still changes with temperature and power limits. Timings therefore move
between runs, and results compare only with care. The simplest route is to measure a model
with `calbredge run` directly, without a project.

### 1. Install

Install in the clone; x86 has no constraints file:

```bash
python3 -m venv ~/calbredge-venv
source ~/calbredge-venv/bin/activate
pip install --upgrade pip      # older pip can't install from a clone
pip install ./protocol "./agent[onnxruntime]"
```

### 2. Check the host

```bash
calbredge doctor
```

It checks what a measurement depends on (the clocks, the power mode, the runtime and its
versions) and prints a fix for anything that fails. It changes nothing and never runs sudo.
On x86 with ONNX Runtime it ends with `Ready.`: the clocks line is a warning (`not
determined`), since no rule shows an x86 CPU's speed held fixed, and the support tier line
says the combination is results only (`docs/SUPPORTED_PLATFORMS.md`).
`calbredge --version` prints the version you installed.

### 3. Run

Measure with the bundled protocol whose input shape is your model's (they are in
`protocol/calbredge_protocol/protocols/`, named for the input size):

```bash
calbredge run --model <your model>.onnx --runtime onnxruntime --protocol protocol/calbredge_protocol/protocols/protocol224.yaml --offline
```

`--offline` guarantees no network calls: the result stays on the host. At any CPU governor
the summary's `Clocks` line says `not determined`, and the result records
`target.clocks_pinned` as `null`: the timings move with the processor's speed, so compare such
results with care, and never with a run whose speeds were locked.

### `calbredge init` on x86

`calbredge init` runs at any CPU governor: it writes the project and measures the model once,
and the result records `target.clocks_pinned` as `null`, like any other x86 run. It prints the
clocks warning and goes on; it stops only where clocks can be pinned and aren't (a Jetson or
an RK3588).

## Reading and comparing results

**The summary.** Every run ends with one:

```text
────────────────────────────────────────────────────
  Calbredge Result Summary
────────────────────────────────────────────────────
  Model       mobilenetv2-12.onnx
  Runtime     tensorrt-fp16
  p50         0.99 ms
  p90         1.12 ms
  p99         1.15 ms
  Throughput  1009.1 img/s
  Status      stable
  Clocks      pinned (jetson)
  hardware_id <hardware_id>
  run_id      <id>
  Saved       ~/.calbredge/results/<id>.json
────────────────────────────────────────────────────
```

That is a Jetson Orin Nano Super in MAXN_SUPER running the example model.

Every run leaves two things on the target:

- `~/.calbredge/results/<id>.json`: the result as the summary shows it.
- `~/.calbredge/runs/<id>/`: the run bundle. `result.json` is the full record,
  `samples.jsonl` has the latency of every measured iteration, `environment.json` the software
  environment, and `logs/` what the run printed. Every field is described in
  `docs/RESULT_SCHEMA.md`.

**Reading one run.** p50, the median, is the number to rely on. In our runs, p50 moved under
2% between runs of the same model; p99 moves several times more, so read a change in p99 as a
hint and not as a result. `Status` is `stable` unless p99 is more than twice the p50
(`inconclusive`) or more than three times (`unstable`); such a run is too noisy to use, so
check nothing else is using the target and run again. `Clocks` should say `pinned`: with
unlocked speeds the timings move with the processor's speed. On x86 it says `not determined`,
because the speeds can't be locked there.

**Comparing two runs.** Calbredge has no compare command: you compare the p50 of two results,
after checking in each bundle's `result.json` that the runs were made the same way. A
difference in any of these fields means the two numbers measure different things:

| Field in `result.json` | What it says |
|---|---|
| `protocol.protocol_digest` | The same protocol: iterations, batch size and input shape |
| `protocol.io_signature` | The model has the same inputs and outputs |
| `candidate.runtime`, `candidate.runtime_version` | The same runtime, at the same version |
| `candidate.precision` | The same precision (FP16 and INT8 are different operating points) |
| `target.hardware_family` | The same kind of target |
| `target.power_mode` | The same power mode |
| `target.clocks_pinned` | `true` in both |

`candidate.model_hash` says whether the model file itself changed, which is usually the change
you are measuring. With all of the above equal, a p50 difference larger than the run-to-run
movement on your target is real; to learn that movement, run the unchanged model three times.

## More

- Several models in one repository: run `calbredge init` in each model's folder; each gets its
  own project.
- Why the speeds are locked, and what a protocol is: `docs/HOW_IT_WORKS.md`.
- The measurement method in full: `docs/PROTOCOL.md`.
- Something went wrong: `docs/TROUBLESHOOTING.md`.
- Every command's options: `calbredge <command> --help`.

`calbredge` and `calbredge-protocol` are licensed under Apache 2.0 (see `LICENSE` and `NOTICE`
in `agent/` and `protocol/`). A separately licensed package, `calbredge-gate`, adds release
gating on top of these measurements; it is not part of this repository.
