# protocol/calbredge_protocol/power_window.py
import re
import subprocess


class TegrastatsWindowSampler:
    """Brackets a measurement window with tegrastats and parses VDD_IN power + RAM.

    Unlike RuntimeAdapter.collect_hardware_metrics (a fixed 2s snapshot taken
    AFTER the loop, so it reads post-inference idle), this starts before the
    measured loop and stops after it, so watts reflect load DURING measurement.
    Mirrors the inline tegrastats bracketing proven in the Track B spike.
    """

    def __init__(self, rail: str = "VDD_IN", interval_ms: int = 100):
        self.rail = rail
        self.interval_ms = interval_ms
        self._proc = None
        self.watts: list[float] = []
        self.ram_mb: list[int] = []

    def start(self):
        try:
            self._proc = subprocess.Popen(  # noqa: S603 — static command name, interval is an int cast to str, no shell, no external input
                ["tegrastats", "--interval", str(self.interval_ms)],  # noqa: S607 — hardcoded arg list, no shell, no user input
                stdout=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            self._proc = None  # degrade to null power, never fail the run
        return self

    def stop(self, timeout: float = 5.0):
        if self._proc is None:
            return self
        self._proc.terminate()
        out, _ = self._proc.communicate(timeout=timeout)
        for line in out.splitlines():
            m = re.search(
                rf"{self.rail} (\d+)mW", line
            )  # first number = instantaneous, matches spike
            if m:
                self.watts.append(int(m.group(1)) / 1e3)
            m = re.search(r"RAM (\d+)/(\d+)MB", line)
            if m:
                self.ram_mb.append(int(m.group(1)))
        return self

    @property
    def watts_avg(self):
        return float(sum(self.watts) / len(self.watts)) if self.watts else None

    @property
    def watts_peak(self):
        return max(self.watts) if self.watts else None

    @property
    def ram_peak_mb(self):
        return max(self.ram_mb) if self.ram_mb else None

    @property
    def energy_joules_window(self):
        # true integral over the sampled window; bonus field, NOT the seam-check
        # target (J/chunk below matches the spike's W*p50 derivation instead).
        return sum(self.watts) * (self.interval_ms / 1e3) if self.watts else None
