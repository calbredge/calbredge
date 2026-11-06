"""External USB power meter support (KM003C) for board-level power capture.

This module gives adapters that lack an on-chip power rail (RK3588 has no
Jetson-style ``tegrastats``/``VDD_IN`` telemetry) a way to attach an external
USB power meter and get real watts instead of shipping ``hardware_metrics``
as permanently ``None``.

Hardware: ChargerLAB KM003C or compatible clone exposing the same USB
interface (VID 0x5fc9, PID 0x0063). The meter sits inline on the 5V USB-C
power input to the board, so everything this reports is **whole-board input
power**, not module/rail power — see ``power_domain="board_input"`` on the
resulting :class:`~calbredge_protocol.models.HardwareMetrics` (set by the adapter, not
here). That is a different measurement point than Jetson's ``tegrastats``
(on-die rail sensors), so cross-hardware power/energy comparisons must key
off ``power_domain`` and ``power_source``, not assume equivalence.

``pyusb`` is an optional dependency (``protocol[power-meter]``) and is
imported lazily inside :meth:`PowerSampler.start`/:meth:`PowerSampler.stop`,
never at module import time — so importing this module, and anything that
imports it (``calbredge_protocol.rknn_adapter``), never requires ``pyusb`` to be
installed. The meter is reference instrumentation only: it is never required
for a benchmark run to succeed. Every failure mode below — ``pyusb`` not
installed, no meter physically attached, or a USB error mid-run — degrades
to "no power data," never to a raised exception out of :meth:`start` or
:meth:`stop`.

Only one process may hold the meter open at a time (this claims the USB
interface exclusively). Never run alongside another tool that reads the same
meter (e.g. a standalone ``record.py`` script) — the two will fight over the
device.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

_VENDOR_ID = 0x5FC9
_PRODUCT_ID = 0x0063
_INTERFACE = 1
_WRITE_ENDPOINT = 0x5
_READ_ENDPOINT = 0x85
_REQUEST_PAYLOAD = bytes([0x0C, 0x0, 0x2, 0x0])
_RESPONSE_LENGTH = 64
_USB_TIMEOUT_MS = 200

# The meter's true internal update rate is ~10Hz (~100ms) — sampling faster
# than that just re-reads the same latched value. 50ms oversamples by 2x,
# which is enough headroom that we never straddle an update boundary and
# under-sample a fast transient, without spamming the bus for no benefit.
#
# Flagged, not silently reconciled: record.py's own comment claims interface
# 1 sustains ~500 samples/sec at the USB/libusb level — much faster than the
# ~10Hz this constant assumes. Both can be true (the bus can be polled fast;
# the meter's internal ADC/display may still only latch a new value ~10x/sec,
# so faster polling just re-reads the same latched bytes) but that's an
# assumption, not something confirmed on real hardware. If the on-device
# smoke test shows genuinely new values arriving faster than 100ms apart,
# lower this to sample closer to that true rate.
_SAMPLE_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True)
class PowerReading:
    """Aggregated result of one bracketed :class:`PowerSampler` window.

    ``watts_average`` is the time-integrated mean (Σ P·dt / Σ dt), not a
    naive per-sample average — with the sampler oversampling a ~10Hz meter,
    a naive mean would silently double-weight samples that just repeat the
    last-read value, which time-integration is immune to.
    """

    watts_average: float
    watts_peak: float
    energy_joules_window: float
    sample_count: int


class PowerSampler:
    """Background-thread sampler for the KM003C USB power meter.

    Usage mirrors a start/stop bracket around exactly the code whose power
    draw you want measured::

        sampler = PowerSampler()
        sampler.start()
        ... do the timed work ...
        reading = sampler.stop()  # None if no meter was available

    Every public method is safe to call whether or not a meter is attached,
    whether or not ``pyusb`` is installed, and whether or not the meter
    raises mid-run — callers never need to guard these calls themselves.
    """

    def __init__(
        self,
        sample_interval_seconds: float = _SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._sample_interval_seconds = sample_interval_seconds
        self._device = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[tuple[float, float]] = []  # (timestamp, watts)
        self._available = False

    def start(self) -> bool:
        """Attempt to attach the meter and begin sampling in a background thread.

        Returns ``True`` if a meter was found and sampling started, ``False``
        otherwise (no ``pyusb``, no meter attached, or a USB error while
        claiming the interface) — never raises.
        """
        try:
            import usb.core
            import usb.util
        except ImportError:
            return False

        try:
            device = usb.core.find(idVendor=_VENDOR_ID, idProduct=_PRODUCT_ID)
            if device is None:
                return False

            # Verified against the real record.py on the target hardware: a
            # kernel driver (e.g. usbhid) is often already attached to one of
            # the meter's interfaces and claim_interface fails ("Resource
            # busy") unless it's detached first. is_kernel_driver_active/
            # detach_kernel_driver are libusb/Linux-only — pyusb raises
            # NotImplementedError on backends that don't support the concept
            # (e.g. Windows); skip detachment there rather than fail start().
            try:
                for cfg in device:
                    for intf in cfg:
                        intf_num = intf.bInterfaceNumber
                        if device.is_kernel_driver_active(intf_num):
                            device.detach_kernel_driver(intf_num)
            except NotImplementedError:
                pass

            device.set_configuration()
            usb.util.claim_interface(device, _INTERFACE)
        except Exception:
            # Covers usb.core.USBError and anything else pyusb/libusb can
            # throw while a device is present but not cleanly claimable
            # (already held by another process, permissions, etc.).
            return False

        self._device = device
        self._available = True
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        if self._device is None:
            return  # pragma: no cover — guarded by start(), never hit in practice
        while not self._stop_event.is_set():
            watts = self._read_watts()
            if watts is not None:
                self._samples.append((time.perf_counter(), watts))
            self._stop_event.wait(self._sample_interval_seconds)

    def _read_watts(self) -> float | None:
        try:
            self._device.write(_WRITE_ENDPOINT, _REQUEST_PAYLOAD, timeout=_USB_TIMEOUT_MS)
            data = self._device.read(_READ_ENDPOINT, _RESPONSE_LENGTH, timeout=_USB_TIMEOUT_MS)
        except Exception:
            # A transient USB read glitch mid-run must not kill the sampler
            # thread or the benchmark — just skip this sample.
            return None
        # Byte offsets/parsing verified against the real record.py on the
        # target hardware: volts is a plain (unsigned) int read using the
        # host's native byte order — record.py uses sys.byteorder rather
        # than a hardcoded endianness, which we match here since the meter
        # and this process are always the same physical machine. Amps is
        # the same 4 bytes reinterpreted as signed two's-complement (what
        # record.py does via ctypes.c_int32) with abs() applied — direction
        # of current flow isn't meaningful for a power draw reading.
        volts = int.from_bytes(bytes(data[8:12]), byteorder=sys.byteorder) / 1e6
        amps = abs(int.from_bytes(bytes(data[12:16]), byteorder=sys.byteorder, signed=True)) / 1e6
        return volts * amps

    def stop(self) -> PowerReading | None:
        """Stop sampling, release the USB interface, and return the aggregate.

        Returns ``None`` if :meth:`start` never got a meter attached, or if
        no samples were collected during the window (window shorter than one
        sample interval, or every read failed). Never raises — release/reset
        failures during teardown are swallowed so a flaky meter never fails
        the benchmark run itself.
        """
        if not self._available:
            return None

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release_device()

        samples = self._samples
        self._samples = []
        self._available = False

        if len(samples) < 2:
            if len(samples) == 1:
                _, watts = samples[0]
                return PowerReading(
                    watts_average=watts,
                    watts_peak=watts,
                    energy_joules_window=0.0,
                    sample_count=1,
                )
            return None

        energy_joules = 0.0
        watts_peak = samples[0][1]
        # itertools.pairwise/zip(strict=) both need 3.10+; this package
        # supports 3.9 (see pyproject.toml). samples/samples[1:] are
        # inherently mismatched by exactly one element, by construction, so
        # there's no silent-truncation risk zip's strict= guards against here.
        for (t0, p0), (t1, _p1) in zip(samples, samples[1:]):  # noqa: B905, RUF007
            dt = t1 - t0
            energy_joules += p0 * dt
            watts_peak = max(watts_peak, p0)
        watts_peak = max(watts_peak, samples[-1][1])

        total_dt = samples[-1][0] - samples[0][0]
        watts_average = energy_joules / total_dt if total_dt > 0 else samples[-1][1]

        return PowerReading(
            watts_average=watts_average,
            watts_peak=watts_peak,
            energy_joules_window=energy_joules,
            sample_count=len(samples),
        )

    def _release_device(self) -> None:
        device = self._device
        self._device = None
        if device is None:
            return
        try:
            import usb.util

            usb.util.release_interface(device, _INTERFACE)
            device.reset()
            usb.util.dispose_resources(device)
        except Exception:  # noqa: S110 — best-effort teardown, must not propagate
            # Repeated runs must never leave the meter locked, but a failure
            # here must also never fail the benchmark that already completed.
            pass
