"""Auto-start local Qwen3 judge server for MMMU/MMBench scoring."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from vision_encoder_eval.mllm.evaluation.judge_config import resolve_judge_profile, uses_api_judge
from vision_encoder_eval.mllm.utils.config import LOGS_ROOT, VTB_ROOT

_JUDGE_PROC: subprocess.Popen | None = None
_JUDGE_LOG_HANDLE: Any | None = None
_STARTED_BY_US = False
_JUDGE_LOG_PATH: str | None = None
# Parallel MMMU/MMBench scoring used to race and start two Qwen3-32B loads on
# the same reserved GPU → CUDA OOM. Serialize startup across threads.
_ENSURE_LOCK = threading.Lock()


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def judge_server_healthy(base_url: str, timeout: float = 2.0) -> bool:
    url = _models_url(base_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _read_log_tail(log_path: str | None, max_bytes: int = 4000) -> str:
    if not log_path or not os.path.isfile(log_path):
        return ""
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _wait_for_judge(
    base_url: str,
    timeout_sec: float = 600.0,
    proc: subprocess.Popen | None = None,
    log_path: str | None = None,
) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if judge_server_healthy(base_url):
            return
        if proc is not None and proc.poll() is not None:
            tail = _read_log_tail(log_path)
            raise RuntimeError(
                f"Judge process exited with code {proc.returncode}:\n{tail[-2000:]}"
            )
        time.sleep(2.0)
    tail = _read_log_tail(log_path)
    msg = f"Judge server did not become ready at {base_url} within {timeout_sec:.0f}s"
    if tail:
        msg += f"\nLog tail ({log_path}):\n{tail[-2000:]}"
    raise RuntimeError(msg)


def _port_from_base_url(base_url: str | None) -> int:
    if not base_url:
        return 8000
    try:
        port = base_url.rstrip("/").split(":")[-1].split("/")[0]
        return int(port) if port.isdigit() else 8000
    except (TypeError, ValueError):
        return 8000


def _kill_judge_server_pids(ports: list[int]) -> int:
    """SIGTERM any leftover ``judge_server --port N`` processes. Returns kill count."""
    import signal

    killed = 0
    seen: set[int] = set()
    for port in ports:
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", f"judge_server.*--port {port}"],
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in out.splitlines():
            text = line.strip()
            if not text:
                continue
            pid = int(text)
            if pid in seen or pid == os.getpid():
                continue
            seen.add(pid)
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except ProcessLookupError:
                pass
    if killed:
        # Give CUDA a moment to reclaim memory before the next training job.
        time.sleep(3.0)
        for pid in list(seen):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return killed


def _stop_judge() -> None:
    global _JUDGE_PROC, _JUDGE_LOG_HANDLE, _STARTED_BY_US, _JUDGE_LOG_PATH
    if _JUDGE_PROC is not None and _JUDGE_PROC.poll() is None:
        _JUDGE_PROC.terminate()
        try:
            _JUDGE_PROC.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _JUDGE_PROC.kill()
            try:
                _JUDGE_PROC.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    _JUDGE_PROC = None
    if _JUDGE_LOG_HANDLE is not None:
        try:
            _JUDGE_LOG_HANDLE.close()
        except OSError:
            pass
    _JUDGE_LOG_HANDLE = None
    _STARTED_BY_US = False
    _JUDGE_LOG_PATH = None


def stop_local_judge(eval_cfg: dict[str, Any] | None = None) -> None:
    """Stop the local judge server so the next train stage gets a clean GPU.

    Must run at the end of each eval job (not only on process exit): ``run.py``
    queues many mllms in one process, and atexit would keep the judge alive
    across jobs (~Qwen3-32B on the last GPU → next pretrain OOM).
    """
    _stop_judge()
    ports = [8000, 8001]
    if eval_cfg is not None:
        profile = resolve_judge_profile(eval_cfg)
        if profile.get("mode") != "api":
            ports.insert(0, _port_from_base_url(str(profile.get("base_url") or "")))
    # Unique, preserve order
    uniq_ports: list[int] = []
    for port in ports:
        if port not in uniq_ports:
            uniq_ports.append(port)
    killed = _kill_judge_server_pids(uniq_ports)
    if killed:
        print(f"Stopped local judge ({killed} process(es)); GPUs released for next job.")
    try:
        from vision_encoder_eval.mllm.evaluation.sharded_judge import stop_sharded_judge_pool

        stop_sharded_judge_pool()
    except Exception:
        pass


def release_eval_gpu_resources(eval_cfg: dict[str, Any] | None = None) -> None:
    """Public cleanup hook for continuous/discrete eval entrypoints."""
    stop_local_judge(eval_cfg)


def ensure_local_judge(eval_cfg: dict[str, Any] | None) -> None:
    """Start transformers judge on a free GPU if not already listening."""
    global _JUDGE_PROC, _JUDGE_LOG_HANDLE, _STARTED_BY_US, _JUDGE_LOG_PATH

    if uses_api_judge(eval_cfg):
        return

    with _ENSURE_LOCK:
        profile = resolve_judge_profile(eval_cfg)
        base_url = str(profile["base_url"])
        model_path = str(profile["model_path"])
        judge_name = str(profile["model"])

        if judge_server_healthy(base_url):
            print(f"Judge already running at {base_url}")
            return

        if not os.path.isfile(os.path.join(model_path, "config.json")):
            raise RuntimeError(
                f"Judge weights missing at {model_path}. "
                f"Run: conda activate VTB && python -m vision_encoder_eval.mllm.utils.downloads judge"
            )

        cfg = eval_cfg or {}
        from vision_encoder_eval.mllm.evaluation.judge_config import parse_judge_gpu_ids

        judge_gpus = parse_judge_gpu_ids(cfg)
        port = base_url.rstrip("/").split(":")[-1].split("/")[0]
        if not port.isdigit():
            port = "8000"

        cmd = [
            sys.executable,
            "-m",
            "vision_encoder_eval.mllm.evaluation.judge_server",
            "--model-path",
            model_path,
            "--port",
            port,
            "--device",
            "cuda:0",
            "--served-model-name",
            judge_name,
        ]
        env = os.environ.copy()
        # Multi-GPU via CUDA_VISIBLE_DEVICES; judge_server may device_map=auto.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(judge_gpus)
        env["PYTHONPATH"] = f"{VTB_ROOT}:{env.get('PYTHONPATH', '')}"
        env.setdefault("OPENAI_API_KEY", "EMPTY")

        log_dir = LOGS_ROOT
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "judge_server.log")

        print(
            f"Starting local judge ({judge_name}) on GPU(s) [{', '.join(judge_gpus)}], "
            f"port {port} ..."
        )
        print("  (judge reserved cards; model infer uses the remaining GPUs)")
        print(f"  Judge log: {log_path}")
        log_handle = open(log_path, "a", encoding="utf-8")
        log_handle.write(
            f"\n=== judge start {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"gpu={','.join(judge_gpus)} port={port} ===\n"
        )
        log_handle.flush()
        _JUDGE_LOG_HANDLE = log_handle
        _JUDGE_PROC = subprocess.Popen(
            cmd,
            env=env,
            cwd=VTB_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _JUDGE_LOG_PATH = log_path
        _STARTED_BY_US = True
        atexit.register(_stop_judge)

        _wait_for_judge(base_url, proc=_JUDGE_PROC, log_path=log_path)
        print(f"Judge ready at {base_url}")

        if _JUDGE_PROC.poll() is not None:
            raise RuntimeError(
                f"Judge process exited early:\n{_read_log_tail(log_path)[-2000:]}"
            )


def needs_judge_server(datasets: list[str], eval_cfg: dict[str, Any] | None = None) -> bool:
    from vision_encoder_eval.mllm.evaluation.judge_config import uses_llm_judge

    if uses_api_judge(eval_cfg):
        return False
    return any(uses_llm_judge(name) for name in datasets)
