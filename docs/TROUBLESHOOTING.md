# Calbredge Troubleshooting Guide

_Problems met on a Jetson Orin Nano Super and an Orange Pi 5 Plus, each with its cause and fix._

**Tested configuration:** JetPack 6.2 / CUDA 12.6 / TensorRT 10.3 / Ubuntu 22.04 / Python 3.10.
Other configurations may work but are untested. Community PRs welcome for other JetPack versions.

---

## Before Filing an Issue

Run `--debug` and paste the output in your GitHub issue:

```bash
calbredge run --model your_model.onnx --debug
```

This prints your Python version, ONNX Runtime version, CUDA provider status, hardware fingerprint, JetPack version, and tegrastats availability in a single pasteable block. Most support questions can be resolved from this output alone.

Also include:
- The full error message and the exact command you ran
- Your JetPack version if running on Jetson (shown in `--debug` output)

---

## Setup: clocks, install and engines

Problems met while setting a target up, each as symptom, cause and fix. `calbredge doctor` and
`calbredge init` exit 7 when the target isn't ready to measure. Which targets are supported:
`docs/SUPPORTED_PLATFORMS.md`.

### Clocks not pinned (`calbredge doctor` FAIL, `calbredge init` exit 7)

**Symptom:** `calbredge doctor` shows `FAIL  clocks pinned  no (jetson): timings would move` and
exits 7; `calbredge init` stops with `clocks pinned: no (jetson). Nothing was written` and exit 7.
A run's summary says `Clocks      not pinned`.

**Cause:** the clocks aren't locked: the boot unit isn't installed, or the power mode was
changed since it ran. Any `sudo nvpmodel -m` unpins the clocks, and starting the unit again
does nothing once it has run; it has to be restarted.

**Fix:** if the boot unit is installed, restart it (this is the fix doctor prints when it
finds the unit):

```bash
sudo systemctl restart calbredge-jetson-clocks.service
```

If it isn't installed, install it as `docs/QUICKSTART.md` shows (the Jetson's step 3), which pins the clocks
now and at every boot; or, until the next reboot only, `sudo jetson_clocks`. Then
`calbredge doctor` should end with `Ready.` On an RK3588 the unit is
`calbredge-rk3588-clocks.service`, and `sudo scripts/performance_mode.sh` pins until the next
reboot.

### pycuda fails to build during install

**Symptom:** the install stops while building pycuda, with `nvcc: not found` or
`Python.h: No such file or directory`.

**Cause:** pycuda compiles from source on a Jetson. It needs `nvcc` (in `/usr/local/cuda/bin`,
which isn't on `PATH` by default) and the Python headers (`python3-dev`).

**Fix:** `sudo apt-get install -y python3-dev`, and put `PATH=/usr/local/cuda/bin:$PATH` in
front of the pip command, as `docs/JETSON_SETUP.md` shows.

### The TensorRT engine is rebuilt unexpectedly

**Symptom:** a run spends 30 to 90 s building an engine when you expected it to reuse one.

**Cause:** an engine is reused only if it was built from exactly the same inputs: the model
file's content, precision, power mode, input shape and TensorRT version. Changing any of them
builds a new engine, by design, so a stale engine is never used. Engines are stored next to the
model (as ignored files) and, if that folder isn't writable, in `~/.calbredge/engine-cache`. In
CI, the checkout step cleans ignored files before every job, so each job builds the engine
again.

**Fix:** nothing is wrong: the build isn't part of the timed measurement. In CI, if the build
time matters, set `clean: false` on the workflow's checkout step so engines survive between
jobs.

---

## 1. Installation Issues

### onnxruntime crashes with "Assertion failed" or core dump

**Error:**
```
cpuid_info warning: Unknown CPU vendor
Assertion '__n < this->size()' failed. Aborted (core dumped)
```

**Cause:** The standard PyPI `onnxruntime` or `onnxruntime-gpu` wheel was installed on Jetson aarch64 Tegra hardware. The standard wheel is compiled for generic aarch64 and does not support Tegra's CPU features: it crashes on `cpuid` probing.

**Fix:**
```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
pip install "numpy<2"
```

> This wheel is the Jetson AI Lab build specifically compiled for JetPack 6 / CUDA 12.6. It is not on PyPI.

---

### NumPy version conflict

**Error:**
```
A module compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Cause:** The Jetson AI Lab `onnxruntime-gpu` wheel was compiled against NumPy 1.x. If NumPy 2.x is installed (which is the current default on PyPI), the binary extension fails to load.

**Fix:**
```bash
pip install "numpy<2"
```

Run this after every pip operation that might pull in a newer NumPy.

---

### `pip install -e .` overwrites the Jetson onnxruntime-gpu wheel

**Cause:** `onnxruntime` was a hard dependency of the `protocol` and `agent` packages in earlier versions. After any `git pull` that touches either package's `pyproject.toml` (in a source checkout), reinstalling the packages may pull the standard PyPI wheel and overwrite the Jetson build.

**Fix: always reinstall in this order on Jetson:**
```bash
pip install --upgrade pip      # Ubuntu 22.04's stock pip can't install from a clone
pip install -e ../protocol/
pip install -e .
pip uninstall onnxruntime -y
pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
pip install "numpy<2"
```

`onnxruntime` is now an optional dependency, so `pip install -e .` will not automatically reinstall it, but if something re-installs it, follow this sequence.

---

### `calbredge` command not found after install

**Fix:** The command is only available inside the virtual environment. Activate it first:

```bash
source ~/calbredge-venv/bin/activate
calbredge --help
```

To make this permanent, add the activation line to your `~/.bashrc`:
```bash
echo 'source ~/calbredge-venv/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

---

## 2. CUDA / GPU Issues

### CUDAExecutionProvider failed to load

**Error:**
```
Failed to load library libonnxruntime_providers_cuda.so
with error: libcublasLt.so.12: cannot open shared object file: No such file or directory
```

**Cause:** The CUDA library directory is not on `LD_LIBRARY_PATH`. ONNX Runtime finds its CUDA provider plugin but that plugin cannot locate the system CUDA libraries at runtime.

**Fix:**
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Then re-run your benchmark. Confirm CUDA is detected:
```bash
calbredge run --model model.onnx --debug
# Look for: available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

### Benchmark runs on CPU instead of GPU (no error shown)

**Cause:** Either `onnxruntime-gpu` is not installed, or the CUDA libraries are not in `LD_LIBRARY_PATH`. ONNX Runtime silently falls back to CPU when CUDA is unavailable.

**Check:** Look for the provider line in benchmark output:
```
provider: CUDAExecutionProvider   ← GPU is being used
provider: CPUExecutionProvider    ← falling back to CPU
```

**Fix:**
1. Confirm the Jetson GPU wheel is installed: `pip show onnxruntime-gpu`
2. If not installed: follow the [onnxruntime crashes](#onnxruntime-crashes-with-assertion-failed-or-core-dump) fix above
3. If installed but still falling back: set `LD_LIBRARY_PATH` as described in the section above

---

## 3. TensorRT Issues

### TensorRT INT8 engine build fails: YOLOv8n

**Error:**
```
[TensorRT] Assertion min_ <= max_ failed at quantizationBase.cpp
```

**Cause:** YOLOv8n's architecture is unusually sensitive to random-noise INT8 calibration data. The activation ranges computed from noise inputs are degenerate, causing TensorRT's quantization pass to assert.

**Fix:** Use `tensorrt-fp16` instead: it delivers near-identical throughput with no reliability issues:
```bash
calbredge run --model yolov8n.onnx --runtime tensorrt-fp16
```

Real calibration data (representative inference inputs) would likely resolve this, but it is not yet implemented in Calbredge.

---

### TensorRT INT8 fails: YOLOv8n-seg

**Error:**
```
[TensorRT] Could not find any implementation for node /model.22/proto/cv3/conv/Conv
```

**Cause:** An operator in YOLOv8n-seg's segmentation head is not supported under INT8 quantization in TensorRT 10.3.

**Fix:** Use `tensorrt-fp16`:
```bash
calbredge run --model yolov8n_seg.onnx --runtime tensorrt-fp16
```

---

### TensorRT INT8 fails: RT-DETR-S

**Error:**
```
[TensorRT] Assertion item.second != nullptr failed.
region should have been removed from Graph::regions
```

**Cause:** RT-DETR's transformer attention graph structure is incompatible with TensorRT's INT8 quantization pass when using random-noise calibration data.

**Fix:** Use `tensorrt-fp16`:
```bash
calbredge run --model rtdetr_s.onnx --runtime tensorrt-fp16
```

---

### TensorRT engine warning: "Using engine plan file across different models of devices"

**Full warning:**
```
[TensorRT] WARNING: Using an engine plan file across different models and/or configurations of a device.
It is recommended to use a separately calibrated engine for each configuration.
```

**Cause:** TensorRT prints this when an engine was built on a different device model or
configuration. Calbredge reuses an engine only when the model file, precision, power mode,
input shape, calibration and TensorRT version all match the ones it was built from, and never
reads an engine file named without a build key, so it does not load an engine built under
another power mode. The board is not part of the key: the warning appears when engine files
were copied from another board.

**Fix:** Delete the copied engine and its build record beside the model; the next run builds
the engine on this board:
```bash
rm model.onnx.trt.fp16.*.engine model.onnx.trt.fp16.*.engine.json
calbredge run --model model.onnx --runtime tensorrt-fp16
```

---

### `This GPU has no fast FP16 support` (or INT8)

**Error:**
```
RuntimeError: This GPU has no fast FP16 support (TensorRT's platform_has_fast_fp16 is not True), so an FP16 engine could run other kernels while the result records fp16. Refusing to build or load it. Run with --runtime tensorrt-fp32 on this GPU.
```

**Cause:** TensorRT reports that this GPU has no fast kernels for the precision asked for. The
FP16 and INT8 build flags only allow those kernels, so on such a GPU the engine may run others,
and the result would record a precision that did not run. Calbredge refuses the run, whether
the engine would be built or loaded from the cache. A Jetson Orin has fast FP16 and INT8.

**Fix:** measure FP32 on this GPU:
```bash
calbredge run --model model.onnx --runtime tensorrt-fp32
```

---

### TensorRT not found / ImportError

**Error:**
```
ModuleNotFoundError: No module named 'tensorrt'
```

**Cause:** TensorRT comes from JetPack, installed system-wide (at
`/usr/lib/python3.10/dist-packages/`), and can't be installed with pip. A venv created without
`--system-site-packages` doesn't see it.

**Fix:** Recreate the venv with that flag and install again, as `docs/JETSON_SETUP.md` shows:
```bash
python3 -m venv --system-site-packages ~/calbredge-venv
```

Then verify:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## 4. Power Mode Issues

### 7W power mode requires a reboot

**This is expected behaviour.** The Jetson Orin Nano Super's 7W mode uses CPU core hotplugging: it physically takes cores 4 and 5 offline. This requires a clean boot; `nvpmodel` will prompt for it:

```bash
sudo nvpmodel -m 3   # switches to 7W; prompts for reboot
sudo reboot
```

After rebooting, confirm the mode:
```bash
sudo nvpmodel -q
# Should show: NV Power Mode: 7W
```

---

### `pthread_setaffinity_np` warnings in 7W mode

**Warning (appears during ONNX Runtime session init):**
```
pthread_setaffinity_np failed for thread: 0, index: 0, mask: {4,}, error code: 22
pthread_setaffinity_np failed for thread: 1, index: 0, mask: {5,}, error code: 22
```

**Cause:** ONNX Runtime tries to pin its thread pool to specific CPU cores at startup. In 7W mode, cores 4 and 5 are offline (hotplugged out), so the affinity call for those cores fails with `EINVAL` (error code 22).

**Impact:** None. The warning fires once during session creation and does not affect benchmark correctness or latency measurements. ONNX Runtime falls back to scheduling on the 4 available cores automatically.

**Action:** Safe to ignore.

---

### `hardware_id` changes between power modes

**Symptom:** The same device shows different `hardware_id` values across benchmark runs at different power modes, causing results to appear as if they came from different machines.

**Cause:** `hardware_id` is derived from `/sys/devices/system/cpu/possible` (every core the
kernel knows about), not from the online CPU count, which varies because 7W mode takes cores
offline. An install from before that rule gives a different id in each mode.

**Fix:** Update Calbredge: `git pull` in the clone and reinstall both packages as
`docs/JETSON_SETUP.md` describes.

Verify:
```bash
calbredge run --model model.onnx --debug
# hardware_id should be identical across MAXN, 25W, 15W, and 7W runs
```

---

## 5. Network / Upload Issues

### Upload fails with DNS resolution error

**Error:**
```
Failed to resolve '<upload host>'
[Errno -2] Name or service not known
```

**Cause:** The Jetson's WiFi connection dropped. This is common when WiFi power management is enabled: the adapter idles aggressively and loses association during long benchmark runs.

**Immediate fix:**
```bash
sudo iwconfig wlP1p1s0 power off
```

Results are saved locally even when upload fails: re-run the same command without `--dry-run` to upload once connectivity is restored.

**Permanent fix:** Install the WiFi power management service so it disables power management automatically on every boot. See [JETSON_SETUP.md](JETSON_SETUP.md) for the full systemd service configuration.

---

### WiFi drops SSH connection during long benchmark runs

**Cause:** Same as above: WiFi power management aggressively idles the adapter, dropping the SSH session mid-run.

**Permanent fix:**
```bash
sudo nano /etc/systemd/system/wifi-power-off.service
```

Add:
```ini
[Unit]
Description=Disable WiFi power management
After=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/sbin/iwconfig wlP1p1s0 power off

[Install]
WantedBy=multi-user.target
```

Enable it:
```bash
sudo systemctl enable wifi-power-off.service
sudo systemctl start wifi-power-off.service
```

Verify after next reboot:
```bash
iwconfig wlP1p1s0 | grep "Power Management"
# Should show: Power Management:off
```

---

## 6. General

### Benchmark results look unexpectedly slow

Work through this checklist before assuming a regression:

1. **Check the active power mode:**
   ```bash
   sudo nvpmodel -q
   ```
   A result run at 7W will be ~3–4× slower than MAXN for transformer models, ~2× for CNN models.

2. **Lock clocks before benchmarking:**
   ```bash
   sudo jetson_clocks
   ```
   Without `jetson_clocks`, the GPU and CPU governors may not have ramped up to their peak frequencies yet.

3. **Confirm the expected runtime provider:**
   Look for `provider: CUDAExecutionProvider` in output. If it shows `CPUExecutionProvider`, the GPU path is not being used: see [section 2](#2-cuda--gpu-issues).

4. **Check for thermal throttling:**
   Inspect `temperature_peak_celsius` and `thermal_throttle_detected` in the result. Throttling is flagged when the peak temperature exceeds 85 °C. Improve airflow or add a fan.

---

### `The protocol's input_shape ... contradicts the model's input`

**Error (ONNX Runtime):**
```
Benchmark failed: The protocol's input_shape [1, 3, 224, 224] contradicts the model's input 'images', [1, 3, 640, 640], at dimensions 2, 3, which the model fixes. Use --protocol <venv>/lib/python3.10/site-packages/calbredge_protocol/protocols/yolo640.yaml, the bundled protocol with the model's shape [1, 3, 640, 640], or another protocol whose input_shape matches the model's static dimensions.
```

**Cause:** the protocol's `input_shape` has another rank than the model's first input, or
differs from one of its static dimensions. ONNX Runtime feeds the model's static dimensions and
takes only the dynamic ones from the protocol, so it refuses rather than run one shape and
record another.

**Fix:** when a bundled protocol has the model's shape, the message names it with its full
path: pass that `--protocol`. It is the protocol `calbredge init` would choose for the model.
Otherwise write a protocol file with the model's shape (`docs/PROTOCOL.md` lists the bundled
ones).

---

### Model file not found

**Error:**
```
Invalid value for '--model': File does not exist: model.onnx
```

**Cause:** The path is relative and `calbredge` is being run from a different working directory.

**Fix:** Use a path relative to where you are running from, or an absolute path:
```bash
# From the directory that holds models/:
calbredge run --model models/yolov8n.onnx --runtime onnxruntime

# Or with an absolute path from anywhere:
calbredge run --model ~/models/yolov8n.onnx --runtime onnxruntime
```

---

### Getting help

Run with `--dry-run --debug` to capture full environment info and paste it in your issue:

```bash
calbredge run --model models/yolov8n.onnx --dry-run --debug
```

Open a GitHub issue with:
- The `--debug` output block
- The full error message and exact command
- Your OS, Python version, and JetPack version (if on Jetson)

## 7. RKNN / Orange Pi Issues

RK3588 with RKNN FP16 is supported and INT8 is results only (`docs/SUPPORTED_PLATFORMS.md`); these notes come from measurements on an Orange Pi 5 Plus. The issues below assume the board and toolkit are already working and cover model-conversion and quantization behavior specific to RKNN.

### RKNN INT8 quantization silently zeros the classification head on stock YOLO exports

**Symptom:** RKNN-Toolkit2 `build(do_quantization=True, ...)` completes without error on a
stock Ultralytics ONNX export (`yolo export format=onnx`). The resulting `.rknn` model loads
and runs on-device without error, reports plausible-looking box coordinates, but every
class-confidence score is exactly zero. No warning, no error: the benchmark looks "stable"
and fast while producing completely unusable detections.

**Cause:** Stock Ultralytics ONNX export fuses the DFL box decode, sigmoid activation,
and per-scale concat into a single `(1, 84, 8400)` output tensor. RKNN's per-tensor INT8
quantization applies one scale/zero-point across that entire fused tensor. Quantizing the
sigmoid output through the same scale as the box-coordinate regression appears to force the
class-confidence values to collapse to zero. (Weight-level outlier values in the `cv3.*`
classification-head layers are also present (`abs_mean ~0.1-0.2` vs outlier values of
`10-17`) but confirmed NOT the primary cause: the same outlier weights are present in
Rockchip's working export and don't cause the failure there. The mechanism is the shared
quantization scale across the fused `Concat` output, not the weight distribution.)

**Confirmed NOT fixed by:**
- More/better INT8 calibration data (tested 1 image → 4 varied images, identical failure)
- Ultralytics' native `format=rknn` export as of v8.4.93 (its "disable end2end" only strips
  NMS/box-decode postprocessing, not the underlying fused sigmoid/DFL structure: same
  33600/705600 nonzero pattern, i.e. only the 4 box channels survive, all 80 class channels
  are exactly zero)

**Fix:** Use Rockchip's reference ONNX export (`rknn_model_zoo/examples/yolov8/model/download_model.sh`)
instead of the stock Ultralytics export: it restructures the graph to expose per-scale
box/class/score-sum tensors separately (9 outputs for YOLOv8n instead of 1), with
sigmoid/decode math moved to ordinary float32 CPU postprocessing rather than being fused
into the quantized graph.

For custom-trained models: use Rockchip's patched Ultralytics fork,
`airockchip/ultralytics_yolov8`, which keeps training identical to stock Ultralytics and
only modifies the export step (see its `RKOPT_README.md`). Not yet independently verified
on a custom-trained model so far.

**Diagnostic check for any future model:** this used to require a manual step: running
inference by hand and checking `np.count_nonzero()` on each output tensor before trusting
benchmark numbers, because there was no automated way to catch a class-confidence branch that
had silently collapsed. That manual step is no longer necessary. `validate_conversion()` in
Calbredge's RKNN conversion check (`calbredge.validate_rknn`) now runs this check automatically as part of every RKNN
conversion: it runs the original ONNX model and the converted RKNN model on the same real
image, compares their outputs in small groups of channels, and fails the conversion if a group
that had a real signal in the original loses almost all of its magnitude after conversion. It
does this without needing to know which channels are "class scores" or anything else about the
model's architecture, so it works the same way whether the model is a YOLO detector, a
classifier, or something custom. Reach for `validate_rknn.py` first; only fall back to a manual
`np.count_nonzero()` check if you are debugging the validator itself.

---

### Fix confirmed on YOLOv8n-seg (segmentation)

Rockchip's reference export graph fix (above) generalizes to yolov8n-seg. Verified via
Rockchip's hosted `yolov8n-seg.onnx` → RKNN INT8: all 13 output tensors (per-scale box/
class-confidence/confidence-sum/mask-coefficients ×3 scales + shared mask-prototype tensor)
fully populated, no degenerate zero branches. Class-confidence tensors specifically:
100% nonzero, values in plausible 0.002-0.88 range (vs. exactly 0.0 on broken exports).

Notably: only ONE outlier weight found in this model's build log (`cv3.1.1.conv.weight`,
-12.310) vs FOUR in plain yolov8n (including model.0.conv.weight at -17.494). Different
model, different outlier profile: reinforces that outlier weights alone are not the root
cause; the export graph structure is what matters.

---

### No Rockchip reference export exists for RT-DETR-S

**Symptom:** The "use Rockchip's reference export" fix above has no equivalent to reach for
on RT-DETR-S.

**Cause:** `rknn_model_zoo` provides a Rockchip-patched reference ONNX export for every YOLO
variant checked, but has no equivalent entry for RT-DETR-S: confirmed absent as of this
writing. Its transformer-based detection head was never restructured into the separated
per-scale box/class tensor layout that fixes the zeroing bug for YOLO models.

**Fix:** No INT8 fix currently exists for RT-DETR-S on RKNN. Use `rknn-fp16` instead: the
same fallback pattern already used for RT-DETR-S's TensorRT INT8 failure (see
[section 3](#3-tensorrt-issues)). Revisit if Rockchip publishes a reference export in the future.

---

### RKNN-Toolkit2 hybrid quantization: documented fallback, not the fix used here

**Cause:** RKNN-Toolkit2 ships a documented hybrid quantization mode
(the `hybrid_quant` example among the rknn-toolkit2 and model zoo repositories' function examples) that lets you mix
per-layer precision: e.g. keep sensitive layers in fp16 while quantizing the rest to int8.
It's a legitimate tool for genuine accuracy-degradation cases.

**Fix:** Not applicable to the classification-head zeroing bug above: that failure is a hard
zero-out caused by a shared quantization scale across a fused tensor, not a
precision-sensitivity problem hybrid quantization is designed to solve. Reach for hybrid
quantization when INT8 accuracy is degraded-but-nonzero; reach for the reference-export fix
(above) when outputs are exactly zero.

---

### Custom / non-YOLO architectures have no guaranteed automatic INT8 fix

**Cause:** The zeroing-bug fix relies on a Rockchip-maintained reference export (or the
`airockchip/ultralytics_yolov8` fork) that restructures the graph to keep box and class
outputs on separate tensors before quantization. Neither exists for arbitrary custom or
non-YOLO architectures. The likely underlying mechanism, per the outlier-weight investigation
above: RKNN's per-tensor INT8 quantization assigns a single scale/zero-point to the output of
a `Concat` node. When that Concat merges tensors with very different value ranges (e.g.
box-regression values vs. small sigmoid-activated class scores), the shared scale is
dominated by the larger-range tensor and collapses the smaller-range branch toward zero.

**Fix:** There is no automatic fix for custom architectures. Benchmark `rknn-fp16` instead of
`rknn-int8` for any custom or non-YOLO model, unless you can manually restructure the export
graph to separate heterogeneous-range outputs before the final `Concat` the way Rockchip's
reference exports do. Try hybrid quantization first if the failure looks like degraded
accuracy rather than a hard zero-out: it does not help with the hard-zero case.

**Detection is no longer YOLO-only, even though fixing still is.** Everything above is about
*fixing* a corrupted conversion: there remains no automatic repair for a custom or non-YOLO
architecture, because the fix depends on a Rockchip-maintained reference export that only
exists for YOLO models. *Detecting* a corrupted conversion is a different problem, and that one
is now solved for any architecture: `validate_rknn.py` (see the diagnostic-check note above)
compares the original and converted model's outputs in small channel groups without needing to
know what those channels mean, so it catches a silent collapse on a custom model exactly the
same way it catches one on YOLOv8n. In practice this means: if `validate_rknn.py` flags a
custom model as corrupted, you should still expect to fall back to `rknn-fp16` (there is no
automatic INT8 fix), but you will now know that you need to, instead of shipping a silently
broken benchmark without realizing it.

---

### Unexplained ~9% latency variance on RK3588 runs

**Symptom:** RK3588 benchmark runs show roughly 9% run-to-run latency variance even after
locking the CPU governors to performance mode: a level of variance that governor settings
alone should have already eliminated.

**Cause:** the RK3588 is a "big.LITTLE" chip, meaning it packs two different types of CPU core
onto the same die: four large, fast "big" cores (ARM Cortex-A76) built for raw speed, and four
smaller, power-efficient "little" cores (ARM Cortex-A55) built to handle light work cheaply.
Before the model can run on the NPU (neural processing unit, the dedicated chip that actually
executes the AI model), a dispatch step on the CPU hands the work off to it, and which core type
the Linux scheduler picked for that dispatch step was not controlled by Calbredge. Some runs
landed on a fast A76 core and some landed on a slow A55 core, and that alone was enough to shift
the whole run's latency, because the dispatch step's speed depends on which core type ran it.
This was confirmed directly: locking one run to only A76 cores and a separate run to only A55
cores produced two cleanly separated latency bands (roughly 20ms on the A76 cores versus
roughly 25-27ms on the A55 cores) rather than one noisy, unexplained distribution.

**Fix:** use the `--cpu-affinity` flag. By default, Calbredge now detects a big.LITTLE CPU like
the RK3588's and automatically pins the NPU-dispatching thread to the big/fast cores, which
removes this source of variance. If you want the unpinned number instead (for example, to
estimate what a real deployment will see if it does not control core placement), pass
`--cpu-affinity none` to disable pinning explicitly.

---

### No power, energy, or temperature data in RK3588/Orange Pi results

**Symptom:** results from an RK3588 have no power draw or energy-per-inference figures
(`hardware_metrics` is null), unlike results from a Jetson (see
[section 4](#4-power-mode-issues)).

**Cause:** the RK3588 has no on-board power telemetry that Calbredge reads. Power on this
board comes from an external meter, a ChargerLAB KM003C inline on the board's 5V USB-C input,
read over USB during the measured iterations. With no meter attached, or without `pyusb`,
the run completes and records no power. Temperature and thermal throttling are not recorded
on an RK3588, with or without a meter.

**Fix:** attach the meter and install the meter's dependency
(`calbredge-protocol[power-meter]`). Only one process may hold the meter. The figures are the
whole board's draw at its power input, so they don't compare with a Jetson's, which are read
at the module's input rail.

---

### Load-time canary false-positives on synthetic noise for sparse detection outputs

**Symptom:** A model that passes both conversion-time checks (`check_detection_presence()` and
`validate_conversion()` in `calbredge.validate_rknn`) and produces correct detections on
real images still fails to load, with the RKNN adapter's load-time canary (the correctness check
that runs automatically inside `load_model()`, described earlier in this section) reporting
near-zero output at one or more detection scales.

**Cause:** The load-time canary originally verified the converted model by feeding it fixed-seed
random noise rather than a real image, on the theory that any healthy model should produce
broadly nonzero output regardless of input content. A controlled on-device test disproved that
theory for sparse output scales: on yolov8m's INT8 (8-bit integer quantization) build, the
class-confidence tensor for the large-object detection scale (the branch responsible for
detecting objects that fill most of the frame) scored a 0.0037 nonzero ratio when fed random
noise, low enough to trip the canary's failure threshold, but reported a healthy 0.9050 maximum
confidence when fed a real photo. Random noise legitimately contains nothing resembling a large
object, so a healthy large-object detection branch correctly reports almost no confidence on it;
the canary was reading that correct, expected behavior as evidence of a broken conversion.

**Fix:** The canary now verifies against a real calibration image automatically whenever one is
available, resolved the same way the conversion-time checks already resolve one: via the
existing `dataset.txt` convention, or an explicit `canary_image_path` argument. When no real
image can be resolved, the canary still runs against random noise as a fallback, but a low score
there now only prints a warning instead of blocking the load: noise is treated as inconclusive
evidence, not proof of failure.

Note the current limitation: real-image verification only supports NHWC (batch, height, width,
channel; channel-last) input shapes. All current YOLO variants are NCHW (batch, channel,
height, width; channel-first), so they still fall back to the noise-based warning path today.
Extending real-image verification to NCHW models is planned.

---

### `Refusing to load <model>.onnx.rknn.<precision>.default`

**Symptom:** `calbredge run` stops before loading the model with
`Refusing to load ...` and one of: `built from a different model than the .onnx beside it`,
`not the one its build record describes`, `do not hash to its build_key`, `built for precision`
or `built for target_platform`.

**Cause:** the `.rknn` has a build record (`<artifact>.json`, written by `calbredge convert`),
and the record does not describe this `.rknn`, this `.onnx` or this run. The usual case is a
model exported again after it was converted: the `.onnx` is new and the `.rknn` is not.

**Fix:** convert the model again on the conversion host with `calbredge convert`, and copy
the `.onnx`, the `.rknn` and the record to the board together.

### `Warning: ... has no build record`

**Symptom:** a run prints `Warning: <artifact> has no build record` and completes; its result
has no `engine_build`.

**Cause:** the `.rknn` was converted before `calbredge convert` wrote build records, or was
copied without its `.json`. Nothing shows which model it was built from.

**Fix:** copy the record beside the `.rknn` if it exists; otherwise convert the model again.

### `calbredge run` rejects a `.rknn` file passed via `--model`

**Symptom:** `calbredge run --model <path>.rknn --runtime rknn-int8` fails immediately with:
```
Configuration error: RKNNAdapter expects a .onnx path but received '.rknn'.
```

**Cause:** The `--model` argument for the RKNN adapter (RKNNAdapter, the runtime adapter in
the `calbredge-protocol` package that loads and runs an RKNN model) is always the ORIGINAL `.onnx` file, never the
converted `.rknn` file directly. Starting from that `.onnx` path, the adapter derives the
location of the converted file itself, by looking for a file named exactly
`<model>.onnx.rknn.<precision>.<power_mode>` (for example, `yolov8s.onnx.rknn.int8.default`)
sitting in the same directory as the `.onnx` file. This naming convention exists so that a
single `.onnx` file can have several converted variants (different precisions such as int8 and
fp16, different power modes) sitting alongside it on disk at the same time without any
ambiguity about which converted file belongs to which variant, and it deliberately mirrors the
same naming pattern Calbredge already uses for TensorRT's cached engine files.

**Fix:** Put the original `.onnx` file on the device, then create a symlink (or rename the
converted file) so its filename matches exactly the pattern the adapter expects:
```bash
ln -s <converted_file>.rknn <model>.onnx.rknn.<precision>.default
```
Then pass `--model <model>.onnx`, never the `.rknn` path, to `calbredge run`.

---

### Rockchip's YOLOv8 `download_model.sh` only fetches the `yolov8n` variant

**Symptom:** Running the Rockchip model zoo's download script for YOLOv8
(`rknn_model_zoo/examples/yolov8/model/download_model.sh`) downloads `yolov8n.onnx` and nothing
else. There is no error and no warning: other model sizes such as `yolov8s.onnx` and
`yolov8m.onnx` are simply never fetched, and the script exits as if it had finished successfully.

**Cause:** The script is hardcoded to a single download URL for the `yolov8n` variant only. It
does not take a model-size argument, even though the same model zoo hosts the same style of
reference export for the other YOLOv8 sizes.

**Fix:** Check the model's `README.md` in the same directory
(`rknn_model_zoo/examples/yolov8/README.md`) for direct download links to the other sizes, then
fetch them manually with `wget` using the same URL pattern the script itself uses, saving each
file under its correct name (`yolov8s.onnx`, `yolov8m.onnx`, and so on) in the same `model/`
directory.

---

### A previously-working RKNN INT8 conversion recipe collapses on a new YOLO model size or variant

**Symptom:** Converting a new YOLO model size or variant to RKNN INT8 (8-bit integer
quantization) produces a silent collapse (caught by `validate_conversion()` or
`check_detection_presence()`; see the diagnostic-check note earlier in this section), even
though the exact same conversion recipe worked without issue for `yolov8n`.

**Cause:** This is the same underlying principle established with `yolov8n` earlier in this
section, generalized to a rule worth checking before every new conversion: a model's STOCK
export (the ONNX file produced straight from the original training framework's own
export tool, for example Ultralytics's `yolo export format=onnx`) can have a fused output
layout that INT8 quantization collapses, while Rockchip's own reference export of the same
architecture (an "unfused" layout, hosted in `rknn_model_zoo`) does not have this problem. This
is a property of the EXPORT (how the ONNX computation graph happens to be structured) and not
a property of the model's trained weights: the same trained model produces two different ONNX
graphs depending on which export tool wrote them, and only one of those graphs quantizes
cleanly.

**Fix:** Before converting any new YOLO-family model, check whether Rockchip's model zoo
(`rknn_model_zoo/examples/`) hosts a matching reference export for that architecture and size,
and use that instead of a stock export from the training framework's own export tool. If no zoo
export exists for that specific model, convert the stock export anyway, but treat the result
with extra scrutiny: confirm both `validate_conversion()` and `check_detection_presence()` pass
before trusting any benchmark numbers produced from it.
