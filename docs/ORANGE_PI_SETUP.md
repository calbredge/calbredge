# Orange Pi 5 Plus Setup Guide

## Tested Configuration

| Component | Version |
|-----------|---------|
| Device | Orange Pi 5 Plus 8 GB (RK3588, the full chip, not the RK3588S) |
| OS Image | Orange Pi official vendor image, Ubuntu Jammy |
| Kernel | 5.10.160 (vendor kernel, ships Rockchip's `rknpu` NPU driver) |
| Conversion host | x86_64, Python 3.12.3, RKNN-Toolkit2 2.3.2, CPU-only `torch` 2.4.0 |
| On-device runtime | `librknnrt.so` 2.3.2 (manually replaces the vendor image's bundled 1.4.0) |
| onnx (Python, conversion host) | 1.18.0 exact |
| Architecture | aarch64 (board), x86_64 (conversion host) |

> **Note:** RKNN-Toolkit2 2.3.2 is confirmed current as of this writing. Only the vendor Ubuntu Jammy image + Toolkit2 2.3.2 combination above is verified; other combinations may work but are untested. Community PRs welcome.

> **Do not use Armbian for this board.** Armbian's default vendor kernel binds the NPU to the mainline `rocket` driver rather than Rockchip's `rknpu`. RKNN cannot run at all on that image, regardless of how RKNN-Toolkit2 or the runtime libraries are installed. See [Common Errors and Fixes](#common-errors-and-fixes) below for what this looks like if you hit it anyway.

---

## Step-by-Step Setup (fresh install)

### 1. Get a USB-C microSD card reader

Most modern laptops have no SD card slot. You'll need a USB-C card reader to flash the microSD card from your conversion host.

### 2. Download the vendor OS image, not Armbian

Download Orange Pi's official vendor image (Ubuntu Jammy, kernel 5.10.160) from [orangepi.org](http://www.orangepi.org/): search the Orange Pi 5 Plus downloads page for the Ubuntu Jammy server or desktop image.

Armbian was tried first during bring-up and rejected for two reasons:
- Its headless first-boot profile (the config that's supposed to apply a static network config and a custom user) silently did not apply: the board still came up on DHCP with the default `root`/`1234` login regardless of what was pre-configured on the SD card.
- More fundamentally, Armbian's default vendor kernel binds the NPU to the mainline `rocket` driver rather than Rockchip's own `rknpu` driver. RKNN inference cannot run at all in that configuration: this isn't a permissions or driver-version problem, it's the wrong driver bound to the device entirely.

The vendor Ubuntu Jammy image ships the correct `rknpu`-based kernel out of the box.

### 3. Flash to microSD

```bash
# On your conversion host, via the USB-C card reader
# Using balenaEtcher (GUI) or:
sudo dd if=Orangepi5plus_x.x.x_ubuntu_jammy_server_arm64.img of=/dev/sdX bs=4M status=progress
sync
```

### 4. First boot

1. Insert the microSD into the Orange Pi 5 Plus.
2. Connect to your network via Ethernet (the board has dual 2.5G ports).
3. Power on.
4. Find the board's IP via your router's DHCP table or `nmap -sn <your subnet>/24`.
5. SSH in with the default credentials: `ssh root@<ip>` (`root`/`1234`). Expect DHCP and default credentials on first boot: don't rely on any pre-seeded network/user config taking effect; set both up manually after logging in.

### 5. Set a static IP (recommended)

```bash
nmcli con mod "Wired connection 1" ipv4.addresses <board-ip>/24
nmcli con mod "Wired connection 1" ipv4.gateway <gateway-ip>
nmcli con mod "Wired connection 1" ipv4.method manual
nmcli con up "Wired connection 1"
```

### 6. Add your user to the `render` group

NPU device access requires membership in the `render` group, **not** just `video`. `video`-only membership will pass some cursory checks but fail with a permission error the first time the NPU is actually initialized.

```bash
sudo usermod -aG render $(whoami)
# Log out and back in (or reboot) for the group change to take effect
groups   # confirm 'render' is listed
```

### 7. Verify the NPU is accessible

```bash
ls /dev/dri/                      # should show renderD128 or similar
cat /sys/kernel/debug/rknpu/load  # shows NPU load, which confirms the rknpu driver is bound
```

If `/sys/kernel/debug/rknpu/load` doesn't exist, the board is not running the vendor kernel's `rknpu` driver: double check you flashed the vendor Ubuntu Jammy image and not Armbian.

### 8. Install system dependencies and a Python virtual environment

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git cmake build-essential \
    libglib2.0-dev libssl-dev wget

python3 -m venv ~/calbredge-venv
source ~/calbredge-venv/bin/activate
```

### 9. Install RKNN-Toolkit2 on the conversion host (x86_64)

Model conversion from ONNX to RKNN happens on x86, not on the board itself. In a clone of the
repository on the conversion host, install the toolkit and the `calbredge` agent's
`rknn-convert` extra (Pillow, for the conversion-time correctness check) with the converter's
constraints file, which holds the versions tested with Python 3.12.3:

```bash
python3 -m venv ~/calbredge-venv
source ~/calbredge-venv/bin/activate
pip install -c constraints/rknn-convert-x86_64-py312.txt --extra-index-url https://download.pytorch.org/whl/cpu rknn-toolkit2 ./protocol "./agent[rknn-convert]"
```

The file pins `onnx` to exactly 1.18.0 (newer releases break the toolkit's `load_onnx`),
`setuptools` below 81 (the toolkit imports `pkg_resources`, which 81 removed) and a `protobuf`
the toolkit's graph optimizer works with under the default implementation. The extra index is
PyTorch's CPU-only build: the toolkit requires `torch` 2.4.0 or older, conversion uses no GPU,
and without the index pip installs the CUDA build: about 2.8 GB of downloads and 5.6 GB on disk,
against about 400 MB of downloads and 1.6 GB on disk with the CPU build.

### 10. Replace the vendor image's `librknnrt.so` with the current version

The vendor OS image ships an outdated `librknnrt.so` (v1.4.0) that does not match RKNN-Toolkit2 2.3.2. Models converted with the current toolkit will not load correctly against the stale runtime library. Replace it manually with the matching v2.3.2 build from the [`airockchip/rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2) repo:

```bash
# On the Orange Pi, after downloading the matching librknnrt.so from the repo's
# rknpu2/runtime/Linux/librknn_api/aarch64/ directory:
sudo cp librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig
```

### 11. Install Calbredge and the RKNN Lite runtime on the Orange Pi

In a clone of the repository, with the venv active. The `rk3588` extra installs
`rknn-toolkit-lite2`, and the constraints file holds the versions validated on this board:

```bash
pip install -c constraints/rk3588-jammy-py310.txt ./protocol "./agent[rk3588]"
```

### 12. Verify the RKNN runtime on the Orange Pi

```bash
python3 -c "from rknnlite.api import RKNNLite; print('RKNN Lite OK')"
```

### 13. Convert and run a first model

On the conversion host:

```bash
python3 -c "
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588', mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
rknn.load_onnx(model='yolov8n.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('yolov8n.rknn.fp16')
rknn.release()
"
```

`mean_values`/`std_values` in `config()` are the input normalization the model expects (here, scale `[0,255]` raw pixel input down to `[0,1]`), not a precision setting. Precision (fp16 vs. int8) is controlled entirely by `do_quantization` in `build()`: `False` for fp16, `True` (with a calibration `dataset`) for int8. There is no `quant_type='none'` or equivalent fp16 flag in `config()`.

Transfer the resulting `.rknn` file to the Orange Pi and run inference to confirm the setup end-to-end. This checks the toolkit only: the file has no build record and not the name Calbredge looks for, so convert a model you mean to measure as step 14 shows.

### 14. Convert a model for benchmarking

On the conversion host, `calbredge convert` builds the `.rknn` under the name the RKNN adapter
looks for, and writes its build record beside it:

```bash
calbredge convert --framework rknn --model yolov8n.onnx --precision fp16 --output yolov8n.onnx.rknn.fp16.default
```

That leaves `yolov8n.onnx.rknn.fp16.default` and `yolov8n.onnx.rknn.fp16.default.json`. The
record names the source `.onnx` by hash, the `.rknn` by hash, the toolkit version, the
precision and target, the model's inputs as declared and as converted, and the calibration
digest for an INT8 build. Copy the `.onnx`, the `.rknn` and the record to the board together:
the record describes one `.rknn` built from one `.onnx`, and converting again replaces both
files. Two conversions of the same model give the same build key but not the same bytes, so a
record never fits a `.rknn` from another conversion.

On the board, `calbredge run` checks the `.rknn` against its record and the `.onnx` before
loading it, and refuses a mismatch with the check that failed. A `.rknn` with no record runs
with a warning, and its result does not show which model it was built from.

---

## Common Errors and Fixes

### RKNN inference fails to initialize / no NPU load ever registers

**Error:**
```
E RKNN: [xx:xx:xx.xxx] failed to submit!, op id: 0, op name: ...
```
or the NPU load stays at 0% under `/sys/kernel/debug/rknpu/load` no matter what runs.

**Cause:** The board is booted on an image (typically Armbian) whose default kernel binds the NPU to the mainline `rocket` driver instead of Rockchip's `rknpu` driver. RKNN cannot run against `rocket` at all: this isn't fixable with permissions or library changes.

**Fix:** Flash the Orange Pi official vendor Ubuntu Jammy image (kernel 5.10.160) instead. Confirm the correct driver is active:
```bash
cat /sys/kernel/debug/rknpu/load
# Presence of this path confirms rknpu is bound
```

---

### Permission denied opening the NPU device despite being in the `video` group

**Error:**
```
E RKNN: rknn_init, error code: RKNN_ERR_DEVICE_UNAVAILABLE
```

**Cause:** NPU device nodes under `/dev/dri/` require `render` group membership specifically. Being in `video` alone is not sufficient.

**Fix:**
```bash
sudo usermod -aG render $(whoami)
```
Log out and back in (or reboot), then confirm:
```bash
groups | grep render
```

---

### Model conversion or inference crashes with a version mismatch / symbol errors against `librknnrt.so`

**Cause:** The vendor OS image ships `librknnrt.so` v1.4.0. Models converted with a newer RKNN-Toolkit2 (2.3.2) are not guaranteed compatible with that old runtime library.

**Fix:** Replace `/usr/lib/librknnrt.so` on the board with the version matching your installed toolkit, downloaded from [`airockchip/rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2):
```bash
sudo cp librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig
```

---

### `rknn.load_onnx()` fails or hangs with a non-1.18.0 `onnx` install

**Error:**
```
AttributeError: ... (varies by onnx version)
```

**Cause:** RKNN-Toolkit2 2.3.2 requires `onnx==1.18.0` exactly. Newer `onnx` releases change internal APIs that `load_onnx` depends on. Installing the toolkit without the constraints file pulls the newest `onnx`.

**Fix:** install from the constraints file, as step 9 shows; in an existing venv, from the clone:
```bash
pip install -c constraints/rknn-convert-x86_64-py312.txt onnx
```

---

### Graph optimizer crashes during conversion with a protobuf-related traceback

**Error:**
```
TypeError: Descriptors cannot be created directly.
```
or similar protobuf internals errors during `rknn.build()` / `rknn.load_onnx()`.

**Cause:** the venv has a `protobuf` other than the one the constraints file pins, which the toolkit's graph optimizer works with under the default implementation.

**Fix:** reinstall `protobuf` from the constraints file, in the clone:
```bash
pip install -c constraints/rknn-convert-x86_64-py312.txt protobuf
```

---

### `calbredge convert` fails with `No module named 'pkg_resources'`

**Error:**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause:** RKNN-Toolkit2 2.3.2 imports `pkg_resources`, which `setuptools` 81 removed. Installing the toolkit without the constraints file pulls the newest `setuptools`.

**Fix:** install `setuptools` from the constraints file, which pins it below 81, in the clone:
```bash
pip install -c constraints/rknn-convert-x86_64-py312.txt setuptools
```

---

### Board comes up on DHCP with default `root`/`1234` login despite a pre-configured headless profile

**Cause:** Neither the Armbian headless first-boot profile nor any similar pre-seeded network/user config reliably applies on first boot for this board. Whatever you configured on the SD card before first boot, plan for the board to actually come up on DHCP with default credentials.

**Fix:** SSH in with `root`/`1234`, then configure networking and change credentials manually as shown in steps 4–5 above. Don't spend time debugging why a pre-seeded profile "didn't take": treat manual post-boot configuration as the expected path.

---

## Related docs

- [JETSON_SETUP.md](JETSON_SETUP.md): equivalent setup guide for the Jetson Orin Nano Super
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): model-conversion and quantization issues specific to RKNN, once the board itself is up and running
- [QUICKSTART.md](QUICKSTART.md): installing Calbredge on the board and measuring a model

## Community PRs

Only the vendor Ubuntu Jammy image + RKNN-Toolkit2 2.3.2 combination above is verified. If you successfully run Calbredge on a different OS image or toolkit version, please open a PR adding your tested configuration to the table at the top of this file.
