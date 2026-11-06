"""Unit tests for protocol/calbredge_protocol/power_meter.py (PowerSampler / PowerReading).

Every test injects a fake ``usb``/``usb.core``/``usb.util`` module set via
sys.modules (mirroring test_rknn_adapter.py's pattern for mocking
``rknnlite.api``) so none of this requires pyusb installed or real hardware —
including the "pyusb not installed" path itself, which is exercised by
deliberately making the import fail rather than by depending on the test
environment's actual package set.

No test exercises the real background-thread timing (sleep-based sampling
loops are inherently flaky under test); the aggregation math in
PowerSampler.stop() is tested directly against a hand-built ``_samples`` list
instead — see ``TestStopAggregation``.
"""

import sys
import types

import pytest
from calbredge_protocol.power_meter import PowerReading, PowerSampler

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeInterface:
    def __init__(self, number):
        self.bInterfaceNumber = number


class _FakeConfig:
    def __init__(self, interface_numbers):
        self._interfaces = [_FakeInterface(n) for n in interface_numbers]

    def __iter__(self):
        return iter(self._interfaces)


class _FakeDevice:
    def __init__(
        self,
        read_payloads=None,
        read_raises=False,
        interface_numbers=(0, 1, 3),
        kernel_driver_active_on=(),
        kernel_driver_check_raises=None,
    ):
        self.configured = False
        self.reset_calls = 0
        self.write_calls: list[tuple] = []
        self._read_payloads = list(read_payloads or [])
        self._read_raises = read_raises
        self._configs = [_FakeConfig(interface_numbers)]
        self.kernel_driver_active_on = set(kernel_driver_active_on)
        self.detach_calls: list[int] = []
        self._kernel_driver_check_raises = kernel_driver_check_raises

    def __iter__(self):
        return iter(self._configs)

    def is_kernel_driver_active(self, interface_number):
        if self._kernel_driver_check_raises is not None:
            raise self._kernel_driver_check_raises
        return interface_number in self.kernel_driver_active_on

    def detach_kernel_driver(self, interface_number):
        self.detach_calls.append(interface_number)
        self.kernel_driver_active_on.discard(interface_number)

    def set_configuration(self):
        self.configured = True

    def write(self, endpoint, data, timeout=None):
        self.write_calls.append((endpoint, bytes(data), timeout))

    def read(self, endpoint, length, timeout=None):
        if self._read_raises:
            raise OSError("simulated USB read failure")
        return self._read_payloads.pop(0)

    def reset(self):
        self.reset_calls += 1


class _RecordingUsbUtil:
    def __init__(self, claim_raises=False):
        self.claim_calls: list[tuple] = []
        self.release_calls: list[tuple] = []
        self.dispose_calls: list = []
        self.claim_raises = claim_raises

    def claim_interface(self, device, interface):
        if self.claim_raises:
            raise RuntimeError("simulated claim failure")
        self.claim_calls.append((device, interface))

    def release_interface(self, device, interface):
        self.release_calls.append((device, interface))

    def dispose_resources(self, device):
        self.dispose_calls.append(device)


def _install_fake_usb(monkeypatch, device, usb_util: _RecordingUsbUtil):
    usb_core_module = types.ModuleType("usb.core")
    usb_core_module.find = lambda **kwargs: device
    usb_util_module = types.ModuleType("usb.util")
    usb_util_module.claim_interface = usb_util.claim_interface
    usb_util_module.release_interface = usb_util.release_interface
    usb_util_module.dispose_resources = usb_util.dispose_resources

    usb_package = types.ModuleType("usb")
    # Set explicitly rather than relying on import machinery to wire the
    # parent->submodule attribute when both are pre-seeded in sys.modules.
    usb_package.core = usb_core_module
    usb_package.util = usb_util_module

    monkeypatch.setitem(sys.modules, "usb", usb_package)
    monkeypatch.setitem(sys.modules, "usb.core", usb_core_module)
    monkeypatch.setitem(sys.modules, "usb.util", usb_util_module)


def _payload(volts_micro: int, amps_micro: int) -> bytes:
    data = bytearray(64)
    data[8:12] = volts_micro.to_bytes(4, "little", signed=True)
    data[12:16] = amps_micro.to_bytes(4, "little", signed=True)
    return bytes(data)


# ---------------------------------------------------------------------------
# start() — every degradation path must return False, never raise
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStart:
    def test_returns_false_when_pyusb_not_installed(self, monkeypatch):
        """sys.modules['usb'] = None is the standard trick to force ImportError
        on `import usb.core` regardless of whether pyusb happens to be
        installed in the test environment."""
        monkeypatch.setitem(sys.modules, "usb", None)
        monkeypatch.setitem(sys.modules, "usb.core", None)

        sampler = PowerSampler()
        assert sampler.start() is False
        assert sampler.stop() is None

    def test_returns_false_when_no_meter_found(self, monkeypatch):
        _install_fake_usb(monkeypatch, device=None, usb_util=_RecordingUsbUtil())

        sampler = PowerSampler()
        assert sampler.start() is False
        assert sampler.stop() is None

    def test_returns_false_when_claim_interface_raises(self, monkeypatch):
        device = _FakeDevice()
        _install_fake_usb(monkeypatch, device=device, usb_util=_RecordingUsbUtil(claim_raises=True))

        sampler = PowerSampler()
        assert sampler.start() is False
        assert sampler.stop() is None

    def test_returns_true_and_starts_thread_when_meter_present(self, monkeypatch):
        device = _FakeDevice(read_payloads=[_payload(5_000_000, 1_000_000)] * 20)
        usb_util = _RecordingUsbUtil()
        _install_fake_usb(monkeypatch, device=device, usb_util=usb_util)

        sampler = PowerSampler(sample_interval_seconds=0.01)
        try:
            assert sampler.start() is True
            assert device.configured is True
            assert usb_util.claim_calls == [(device, 1)]
        finally:
            sampler.stop()

    def test_detaches_active_kernel_driver_before_claiming(self, monkeypatch):
        """Verified against the real record.py: claim_interface fails on real
        hardware unless a kernel driver already attached to the interface is
        detached first."""
        device = _FakeDevice(
            read_payloads=[_payload(5_000_000, 1_000_000)] * 20,
            kernel_driver_active_on={1},
        )
        usb_util = _RecordingUsbUtil()
        _install_fake_usb(monkeypatch, device=device, usb_util=usb_util)

        sampler = PowerSampler(sample_interval_seconds=0.01)
        try:
            assert sampler.start() is True
            assert device.detach_calls == [1]
            assert usb_util.claim_calls == [(device, 1)]
        finally:
            sampler.stop()

    def test_kernel_driver_check_not_implemented_still_claims(self, monkeypatch):
        """is_kernel_driver_active/detach_kernel_driver are libusb/Linux-only;
        pyusb raises NotImplementedError on backends without the concept
        (e.g. Windows). That must not block claiming the interface there."""
        device = _FakeDevice(
            read_payloads=[_payload(5_000_000, 1_000_000)] * 20,
            kernel_driver_check_raises=NotImplementedError(),
        )
        usb_util = _RecordingUsbUtil()
        _install_fake_usb(monkeypatch, device=device, usb_util=usb_util)

        sampler = PowerSampler(sample_interval_seconds=0.01)
        try:
            assert sampler.start() is True
            assert usb_util.claim_calls == [(device, 1)]
        finally:
            sampler.stop()


# ---------------------------------------------------------------------------
# _read_watts() — byte-offset parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadWatts:
    def test_parses_volts_and_amps_from_correct_offsets(self):
        sampler = PowerSampler()
        sampler._device = _FakeDevice(read_payloads=[_payload(5_000_000, 2_000_000)])

        watts = sampler._read_watts()

        # 5.0V * 2.0A = 10.0W
        assert watts == pytest.approx(10.0)

    def test_amps_takes_absolute_value_of_negative_reading(self):
        sampler = PowerSampler()
        sampler._device = _FakeDevice(read_payloads=[_payload(5_000_000, -2_000_000)])

        watts = sampler._read_watts()

        assert watts == pytest.approx(10.0)

    def test_returns_none_on_usb_read_error(self):
        sampler = PowerSampler()
        sampler._device = _FakeDevice(read_raises=True)

        assert sampler._read_watts() is None


# ---------------------------------------------------------------------------
# stop() — time-integrated aggregation (whitebox: samples set directly so
# the math is tested without depending on real thread/sleep timing)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStopAggregation:
    def test_without_start_returns_none(self):
        assert PowerSampler().stop() is None

    def test_time_integrated_average_not_naive_mean(self, monkeypatch):
        device = _FakeDevice()
        usb_util = _RecordingUsbUtil()
        _install_fake_usb(monkeypatch, device=device, usb_util=usb_util)

        sampler = PowerSampler()
        sampler._available = True
        sampler._device = device
        sampler._thread = None
        sampler._samples = [(0.0, 2.0), (0.1, 4.0), (0.3, 6.0)]

        reading = sampler.stop()

        # energy = 2.0*0.1 + 4.0*0.2 = 0.2 + 0.8 = 1.0 J
        # total_dt = 0.3s -> watts_average = 1.0/0.3
        assert reading == PowerReading(
            watts_average=pytest.approx(1.0 / 0.3),
            watts_peak=6.0,
            energy_joules_window=pytest.approx(1.0),
            sample_count=3,
        )
        # A naive mean of (2.0, 4.0, 6.0) would be 4.0 — must not equal that,
        # since these samples are unevenly spaced in time.
        assert reading.watts_average != pytest.approx(4.0)

    def test_single_sample_returns_reading_with_zero_window_energy(self, monkeypatch):
        device = _FakeDevice()
        _install_fake_usb(monkeypatch, device=device, usb_util=_RecordingUsbUtil())

        sampler = PowerSampler()
        sampler._available = True
        sampler._device = device
        sampler._thread = None
        sampler._samples = [(0.0, 7.5)]

        reading = sampler.stop()

        assert reading == PowerReading(
            watts_average=7.5, watts_peak=7.5, energy_joules_window=0.0, sample_count=1
        )

    def test_zero_samples_returns_none(self, monkeypatch):
        device = _FakeDevice()
        _install_fake_usb(monkeypatch, device=device, usb_util=_RecordingUsbUtil())

        sampler = PowerSampler()
        sampler._available = True
        sampler._device = device
        sampler._thread = None
        sampler._samples = []

        assert sampler.stop() is None

    def test_stop_releases_and_resets_device(self, monkeypatch):
        device = _FakeDevice()
        usb_util = _RecordingUsbUtil()
        _install_fake_usb(monkeypatch, device=device, usb_util=usb_util)

        sampler = PowerSampler()
        sampler._available = True
        sampler._device = device
        sampler._thread = None
        sampler._samples = [(0.0, 1.0), (0.1, 1.0)]

        sampler.stop()

        assert usb_util.release_calls == [(device, 1)]
        assert device.reset_calls == 1
        assert usb_util.dispose_calls == [device]

    def test_stop_is_safe_when_release_raises(self, monkeypatch):
        """Teardown failures must never propagate out of stop() — a flaky
        meter on disconnect must not fail an otherwise-successful benchmark."""
        device = _FakeDevice()
        usb_util = _RecordingUsbUtil()

        def _raising_release(*args, **kwargs):
            raise RuntimeError("simulated release failure")

        usb_util.release_interface = _raising_release
        _install_fake_usb(monkeypatch, device=device, usb_util=usb_util)

        sampler = PowerSampler()
        sampler._available = True
        sampler._device = device
        sampler._thread = None
        sampler._samples = [(0.0, 1.0), (0.1, 1.0)]

        reading = sampler.stop()  # must not raise

        assert reading is not None
