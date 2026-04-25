"""
Background resource sampler — runs in a thread, samples GPU + per-container
CPU/RAM at 1 Hz, writes JSONL.

Why subprocess + nvidia-smi instead of pynvml:
- pynvml fails inside Docker on WSL2 ("GPU access blocked")
- Subprocess to the host nvidia-smi works reliably from WSL2 Linux
- docker stats --no-stream is the canonical way to read container metrics

Per-container GPU memory attribution is not measurable on consumer GPUs
(no MIG support). GPU numbers are therefore aggregate; per-engine cost is
derived by subtracting an idle baseline from the loaded sample.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _docker_bin() -> str | None:
    """Pick the working docker binary on this box. WSL has both `docker` (broken stub) and `docker.exe`."""
    for name in ("docker", "docker.exe"):
        path = shutil.which(name)
        if path:
            try:
                r = subprocess.run([name, "version"], capture_output=True, timeout=3)
                if r.returncode == 0:
                    return name
            except Exception:
                continue
    return None


def _nvidia_smi_bin() -> str | None:
    for path in (
        "/usr/lib/wsl/lib/nvidia-smi",
        "/mnt/c/Windows/System32/nvidia-smi.exe",
        "nvidia-smi",
    ):
        if shutil.which(path) or Path(path).exists():
            return path
    return None


def _parse_mem(mem_str: str) -> float:
    """Parse '328.6MiB' / '1.5GiB' / '500B' to MB."""
    used = mem_str.split(" / ")[0].strip()
    try:
        if used.endswith("MiB"):
            return float(used[:-3])
        if used.endswith("GiB"):
            return float(used[:-3]) * 1024
        if used.endswith("KiB"):
            return float(used[:-3]) / 1024
        if used.endswith("B"):
            return float(used[:-1]) / (1024 * 1024)
    except ValueError:
        pass
    return 0.0


@dataclass
class Sample:
    timestamp: float                          # time.monotonic()
    gpu: dict[str, Any] = field(default_factory=dict)
    containers: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ResourceSummary:
    samples: int
    gpu_mem_used_mb_peak: float | None
    gpu_mem_used_mb_p50: float | None
    gpu_util_percent_peak: float | None
    gpu_util_percent_p50: float | None
    gpu_temp_c_peak: float | None
    container_cpu_peak: dict[str, float] = field(default_factory=dict)
    container_mem_mb_peak: dict[str, float] = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class ResourceSampler:
    """
    Threaded sampler. Use as a context manager:
        with ResourceSampler(out_path, ['sm-triton', 'sm-rt-transcriber']) as s:
            ... do work ...
        s.summary()  # peak/p50/p95
    """

    def __init__(
        self,
        out_path: Path,
        containers: list[str] | None = None,
        interval_s: float = 1.0,
    ):
        self.out_path = out_path
        self.containers = containers or []
        self.interval_s = interval_s
        self.docker = _docker_bin()
        self.nvsmi = _nvidia_smi_bin()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[Sample] = []

    def _sample_gpu(self) -> dict[str, Any]:
        if not self.nvsmi:
            return {}
        try:
            r = subprocess.run(
                [
                    self.nvsmi, "-i", "0",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode != 0:
                return {}
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            return {
                "mem_used_mb": int(parts[0].split()[0]),
                "mem_total_mb": int(parts[1].split()[0]),
                "util_percent": int(parts[2].split()[0]) if "%" in parts[2] else int(parts[2].split()[0]),
                "mem_util_percent": int(parts[3].split()[0]) if "%" in parts[3] else int(parts[3].split()[0]),
                "temp_c": int(parts[4]),
            }
        except Exception:
            return {}

    def _sample_container(self, name: str) -> dict[str, Any]:
        if not self.docker:
            return {}
        try:
            r = subprocess.run(
                [self.docker, "stats", "--no-stream", "--format", "{{json .}}", name],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode != 0 or not r.stdout.strip():
                return {}
            d = json.loads(r.stdout.strip().splitlines()[0])
            return {
                "cpu_percent": float(d.get("CPUPerc", "0%").rstrip("%")),
                "mem_usage_mb": _parse_mem(d.get("MemUsage", "")),
                "mem_percent": float(d.get("MemPerc", "0%").rstrip("%")),
                "pids": int(d.get("PIDs", 0)),
            }
        except Exception:
            return {}

    def _loop(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            while not self._stop.is_set():
                t = time.monotonic()
                sample = Sample(
                    timestamp=t,
                    gpu=self._sample_gpu(),
                    containers={c: self._sample_container(c) for c in self.containers},
                )
                self._samples.append(sample)
                f.write(json.dumps({
                    "t": sample.timestamp,
                    "gpu": sample.gpu,
                    "containers": sample.containers,
                }) + "\n")
                f.flush()
                # Sleep with early-exit responsiveness
                self._stop.wait(self.interval_s)

    def __enter__(self) -> "ResourceSampler":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self, skip_initial_seconds: float = 30.0) -> ResourceSummary:
        if not self._samples:
            return ResourceSummary(0, None, None, None, None, None)
        t0 = self._samples[0].timestamp
        usable = [s for s in self._samples if s.timestamp - t0 >= skip_initial_seconds] or self._samples

        gpu_mem = [s.gpu.get("mem_used_mb") for s in usable if s.gpu.get("mem_used_mb") is not None]
        gpu_util = [s.gpu.get("util_percent") for s in usable if s.gpu.get("util_percent") is not None]
        gpu_temp = [s.gpu.get("temp_c") for s in usable if s.gpu.get("temp_c") is not None]

        ccpu_peak: dict[str, float] = {}
        cmem_peak: dict[str, float] = {}
        for c in self.containers:
            cpu_vals = [s.containers.get(c, {}).get("cpu_percent") for s in usable]
            cpu_vals = [v for v in cpu_vals if v is not None]
            mem_vals = [s.containers.get(c, {}).get("mem_usage_mb") for s in usable]
            mem_vals = [v for v in mem_vals if v is not None]
            if cpu_vals:
                ccpu_peak[c] = max(cpu_vals)
            if mem_vals:
                cmem_peak[c] = max(mem_vals)

        return ResourceSummary(
            samples=len(usable),
            gpu_mem_used_mb_peak=max(gpu_mem) if gpu_mem else None,
            gpu_mem_used_mb_p50=_percentile(gpu_mem, 50) if gpu_mem else None,
            gpu_util_percent_peak=max(gpu_util) if gpu_util else None,
            gpu_util_percent_p50=_percentile(gpu_util, 50) if gpu_util else None,
            gpu_temp_c_peak=max(gpu_temp) if gpu_temp else None,
            container_cpu_peak=ccpu_peak,
            container_mem_mb_peak=cmem_peak,
        )
