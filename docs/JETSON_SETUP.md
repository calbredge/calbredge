# Jetson Setup Guide

The shortest path from a fresh Jetson to a first measurement is `docs/QUICKSTART.md`. This
page adds the details: what's validated, installing, the optional ONNX Runtime GPU build, engines, power modes, and common errors.

## Validated configuration

| Component | Version |
|---|---|
| Device | Jetson Orin Nano Super 8 GB |
| JetPack | 6.2 (L4T 36.4) |
| CUDA | 12.6 |
| Python | 3.10 (JetPack's default) |
| TensorRT | 10.3.0 (from JetPack) |
| numpy | 1.26.4 (JetPack's TensorRT and onnxruntime-gpu need numpy 1.x) |
| pycuda | 2026.1 (built from source) |
| onnxruntime-gpu | 1.23.0, Jetson AI Lab build (optional) |

Every pinned version is in `constraints/jetson-jp6.2-py310.txt`. Other JetPack versions may
work but aren't validated.

## Installing

`docs/QUICKSTART.md` step 2 has the commands. Two rules make or break a Jetson install: create
the venv with `--system-site-packages` (TensorRT comes from JetPack and can't be installed with
pip), and install with the constraints file (it keeps numpy at 1.x; pycuda compiles from
source, so `nvcc` must be on `PATH` and `python3-dev` installed):

```bash
sudo apt-get install -y python3-venv python3-dev
python3 -m venv --system-site-packages ~/calbredge-venv
source ~/calbredge-venv/bin/activate
PATH=/usr/local/cuda/bin:$PATH pip install -c constraints/jetson-jp6.2-py310.txt ./protocol "./agent[jetson]"
```

Run the last command in a clone of the repository, and activate the venv in every shell.

Check with `calbredge doctor`: it reports TensorRT, pycuda, numpy, the clocks and everything
else a run depends on, with a fix for anything that fails.

**Don't** install the `onnxruntime` extra on a Jetson (`calbredge[onnxruntime]`). It pulls the
x86-oriented `onnxruntime` wheel from PyPI, which can replace the Jetson GPU build and crash on
Tegra hardware.

## ONNX Runtime with the GPU (optional)

TensorRT is the supported runtime on Jetson. ONNX Runtime's CUDA path runs and records results,
but is *results only*: this path isn't validated on a Jetson yet
(`docs/SUPPORTED_PLATFORMS.md`). To
measure with it, install NVIDIA's Jetson AI Lab build into the venv, after Calbredge:

```bash
pip install -c constraints/jetson-jp6.2-py310.txt -i https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.23.0
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## TensorRT engines

TensorRT compiles each model into an engine for this board. The first run for a model builds
it (30 to 90 s); later runs reuse it. An engine is reused only if it was built from exactly the
same model file, precision, power mode, input shape, calibration and TensorRT version, so a
retrained model saved over the same `.onnx` path always gets a new engine.

Engines are written next to the model, or to `~/.calbredge/engine-cache/` if that directory
isn't writable:

```
mobilenetv2-12.onnx.trt.fp16.maxn_super.<16 hex characters>.engine
mobilenetv2-12.onnx.trt.fp16.maxn_super.<16 hex characters>.engine.json   (its build record)
```

`calbredge doctor --model <model.onnx>` says whether a usable engine is already cached. Engine
files without the hex build key come from older Calbredge versions and are never read.

## Power modes and clocks

Power mode numbers differ between boards. Read them from the board, never from another
board's documentation:

```bash
grep 'POWER_MODEL ID=' /etc/nvpmodel.conf
sudo nvpmodel -q
```

On the Orin Nano Super the modes are 15W, 25W (the default), MAXN_SUPER and 7W. Every result
records the mode it ran in, and runs compare only within one mode. Switching
to a mode that takes CPU cores offline (7W) needs a reboot; `nvpmodel` asks when it does.

Pin the clocks before measuring, and after every reboot:

```bash
sudo jetson_clocks
```

Without pinning, the CPU and GPU change speed during a run and the timings move, so a pinned
run never compares with an unpinned one. To pin at every boot, install the unit
`examples/systemd/calbredge-jetson-clocks.service` as `docs/QUICKSTART.md` shows; after a
change of power mode, `sudo systemctl restart calbredge-jetson-clocks.service` pins them again.

## Disable WiFi power management (optional)

The WiFi adapter's power management can drop SSH sessions mid-benchmark. To turn it off for
good:

```bash
sudo nano /etc/systemd/system/wifi-power-off.service
```

```
[Unit]
Description=Disable WiFi power management
After=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/sbin/iwconfig wlP1p1s0 power off
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now wifi-power-off.service
iwconfig wlP1p1s0 | grep "Power Management"   # Expected: Power Management:off
```

Replace `wlP1p1s0` with your interface name (see `iwconfig`).

## Common errors

**`A module compiled using NumPy 1.x cannot be run in NumPy 2.x`.** numpy 2 got in. Reinstall
with the constraints file, or `pip install "numpy<2"`.

**`pycuda` fails to build.** `nvcc` isn't on `PATH` or the Python headers are missing: install
`python3-dev` and run the install with `PATH=/usr/local/cuda/bin:$PATH`.

**`import tensorrt` fails inside the venv.** The venv was created without
`--system-site-packages`. Recreate it with that flag.

**`ImportError: libcublasLt.so.12: cannot open shared object file`.** Add JetPack's CUDA
libraries to the library path:

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**`CUDAExecutionProvider` not available (ONNX Runtime).** The x86 PyPI wheel is installed
instead of the Jetson build. `pip uninstall -y onnxruntime onnxruntime-gpu`, then install the
Jetson AI Lab build as above.

**`malloc(): invalid size (unsorted)` in 15W mode.** Seen under memory pressure in low-power
modes; measure in a higher power mode.

**`tegra` not found in the device tree, or wrong hardware detected, in a container.** Run on
the host, or bind-mount `/proc/device-tree` read-only into the container.

**Python version mismatch.** JetPack 6.2's Python is 3.10, which the pins and the Jetson AI Lab
wheels are built for. Create the venv with `python3.10` if another version is the default.
