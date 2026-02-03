import time
import threading
from typing import Optional

import pynvml


class GPUMaxMemSampler:
    """
    Lightweight NVML sampler that tracks peak GPU memory (bytes).

    Key properties:
    - No globals
    - One sampler per process (recommended, e.g. one Slurm rank / GPU)
    - Low overhead: sleeps between polls (Event.wait does NOT busy-wait)
    - Adaptive interval: polls fast when memory is changing, backs off when stable
    - Optional burst mode at startup to catch early allocations
    - Optional NVML shutdown (OFF by default to avoid interfering with other NVML users)
    """

    def __init__(
        self,
        *,
        gpu_index: Optional[int] = None,
        pci_bus_id: Optional[str] = None,
        min_interval_s: float = 0.01,
        max_interval_s: float = 0.20,
        stable_polls_to_backoff: int = 10,
        backoff: float = 1.4,
        eps_bytes: int = 10 << 20,      # treat +10MiB as "meaningful increase"
        burst_seconds: float = 5.0,    # initial busy phase duration
        burst_interval_s: float = 0.005,
        shutdown_nvml: bool = False,   # safer default: do not shut down globally
    ) -> None:
        if gpu_index is not None and pci_bus_id is not None:
            raise ValueError("Specify only one of gpu_index or pci_bus_id.")

        self._gpu_index = gpu_index
        self._pci_bus_id = pci_bus_id

        self.min_interval_s = float(min_interval_s)
        self.max_interval_s = float(max_interval_s)
        self.stable_polls_to_backoff = int(stable_polls_to_backoff)
        self.backoff = float(backoff)
        self.eps_bytes = int(eps_bytes)

        self.burst_seconds = float(burst_seconds)
        self.burst_interval_s = float(burst_interval_s)

        self.shutdown_nvml = bool(shutdown_nvml)

        # internal state
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._running = False
        self._t0 = 0.0

        self._lock = threading.Lock()
        self._max_used = 0
        self._last_used = 0
        self._samples = 0

    @staticmethod
    def _normalize_pci_bus_id(pci_bus_id: str) -> str:
        """
        Normalize common PCI bus id formats:
          - "BB:DD.F"
          - "0000:BB:DD.F"
          - "00000000:BB:DD.F"  (NVML commonly accepts/returns this)
        """
        s = pci_bus_id.strip()

        # "BB:DD.F" -> "00000000:BB:DD.F"
        if len(s) == 7:
            return "00000000:" + s

        # "0000:BB:DD.F" -> "00000000:BB:DD.F"
        if len(s) == 12 and s[4] == ":":
            return "00000000:" + s[5:]

        # otherwise, leave as-is
        return s

    def _get_handle(self):
        if self._pci_bus_id is not None:
            return pynvml.nvmlDeviceGetHandleByPciBusId(
                self._normalize_pci_bus_id(self._pci_bus_id)
            )
        idx = 0 if self._gpu_index is None else int(self._gpu_index)
        return pynvml.nvmlDeviceGetHandleByIndex(idx)

    def start(self) -> None:
        """Start background sampling."""
        if self._running:
            return

        pynvml.nvmlInit()
        self._handle = self._get_handle()
        if self._handle is None:
            raise RuntimeError("Failed to obtain NVML device handle.")

        with self._lock:
            self._max_used = 0
            self._last_used = 0
            self._samples = 0

        self._t0 = time.time()
        self._stop_evt.clear()
        self._running = True

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling."""
        if not self._running:
            return

        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._running = False
        self._handle = None

        # Safer default is to NOT shutdown NVML (it is global state).
        if self.shutdown_nvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def _run(self) -> None:
        interval = self.min_interval_s
        stable = 0

        while not self._stop_evt.is_set():
            try:
                used = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            except Exception:
                # If NVML hiccups, don't spin. Back off briefly.
                self._stop_evt.wait(0.5)
                continue

            now = time.time()
            in_burst = (now - self._t0) < self.burst_seconds

            # Burst mode: poll at burst_interval (or faster if min_interval is smaller)
            if in_burst:
                interval = min(self.burst_interval_s, self.min_interval_s)
            # else keep current interval (adaptive logic below adjusts it)

            with self._lock:
                self._samples += 1
                if used > self._max_used:
                    self._max_used = used
                prev = self._last_used
                self._last_used = used

            # Adapt polling rate:
            # - if meaningful increase: go fast
            # - if stable for a while: back off (outside burst)
            if used > prev + self.eps_bytes:
                stable = 0
                interval = self.min_interval_s
            else:
                stable += 1
                if stable >= self.stable_polls_to_backoff and not in_burst:
                    stable = 0
                    interval = min(self.max_interval_s, interval * self.backoff)

            # Sleep without busy-waiting
            self._stop_evt.wait(interval)

    @property
    def max_used_bytes(self) -> int:
        """Peak used bytes observed since start()."""
        with self._lock:
            return int(self._max_used)

    @property
    def max_used_mib(self) -> float:
        """Peak used MiB observed since start()."""
        return self.max_used_bytes / (1024.0 * 1024.0)

    @property
    def samples(self) -> int:
        """Number of samples taken since start()."""
        with self._lock:
            return int(self._samples)