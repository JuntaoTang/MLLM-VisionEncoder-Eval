"""Best-effort GPU VRAM reclaim after OOM / failed training attempts."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import os
import subprocess
import time
from typing import Callable, Optional


def nvidia_used_mib(gpu_index: int) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def gpu_compute_pids(gpu_index: int) -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.upper() == "[N/A]":
            continue
        try:
            pids.append(int(line.split()[0]))
        except ValueError:
            continue
    return pids


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def reclaim_gpus(
    gpu_indices: list[int] | None = None,
    *,
    allow_cmd_substrings: tuple[str, ...] = (
        "_probe_micro_batch",
        "probe_micro_batch",
        "_probe_oom_stress",
        "/train_mem.py",
        "torch.distributed.run",
        "torchrun",
    ),
    deny_if_missing_probe_and_has: tuple[str, ...] = (asset_path('trained', 'continuous/'),),
    require_any: tuple[str, ...] = (),
    free_below_mib: float = 2048.0,
    timeout_s: float = 120.0,
    log: Callable[[str], None] = _default_log,
) -> None:
    """Kill matching leftover train processes on the given GPUs, then wait for free VRAM.

    After CUDA OOM, torchrun workers / zombie contexts often keep VRAM occupied so the
    next retry OOMs immediately. Always call this before retrying.
    """
    if gpu_indices is None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            gpu_indices = [int(x.strip()) for x in out.splitlines() if x.strip()]
        except Exception:
            gpu_indices = list(range(8))

    killed: set[int] = set()
    for gpu in gpu_indices:
        for pid in gpu_compute_pids(gpu):
            if pid in killed:
                continue
            cmd = cmdline(pid)
            if not cmd:
                # Dead/zombie CUDA holder — try SIGKILL anyway.
                try:
                    os.kill(pid, 9)
                    killed.add(pid)
                    log(f"[reclaim gpu{gpu}] SIGKILL zombie pid={pid}")
                except Exception:
                    pass
                continue
            if require_any and not any(s in cmd for s in require_any):
                continue
            if not any(s in cmd for s in allow_cmd_substrings):
                continue
            if "_probe_micro_batch" not in cmd and "_probe_oom_stress" not in cmd and any(s in cmd for s in deny_if_missing_probe_and_has):
                # Protect live production trainings unless caller overrides require_any.
                if not require_any:
                    continue
            try:
                os.kill(pid, 9)
                killed.add(pid)
                log(f"[reclaim gpu{gpu}] SIGKILL pid={pid}")
            except ProcessLookupError:
                pass
            except Exception as exc:
                log(f"[reclaim gpu{gpu}] kill pid={pid} failed: {exc}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = [nvidia_used_mib(g) for g in gpu_indices]
        if all(u is None or u < free_below_mib for u in used):
            log(f"[reclaim] ok used_mib={used}")
            return
        time.sleep(1.0)
    used = [nvidia_used_mib(g) for g in gpu_indices]
    log(f"[reclaim] timeout used_mib={used} — continuing")
