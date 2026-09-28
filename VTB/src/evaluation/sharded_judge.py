"""Multi-GPU local judge pool: one Qwen3-8B server per GPU, round-robin at scoring time."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Iterator

from src.evaluation.judge_config import resolve_judge_profile, uses_api_judge
from src.evaluation.judge_manager import judge_server_healthy
from src.utils.config import LOGS_ROOT, VTB_ROOT

_POOL: "ShardedJudgePool | None" = None


class RoundRobinJudge:
    """Dispatch judge API calls across multiple OpenAI-compatible backends."""

    is_api = True

    def __init__(self, judges: list[Any]):
        if not judges:
            raise ValueError("RoundRobinJudge requires at least one backend")
        self._judges = judges
        self._lock = threading.Lock()
        self._cursor = 0
        primary = judges[0]
        self.fail_msg = getattr(primary, "fail_msg", "Failed to obtain answer via API. ")
        self.verbose = getattr(primary, "verbose", False)
        self.default_kwargs = getattr(primary, "default_kwargs", {})

    def _next(self) -> Any:
        with self._lock:
            judge = self._judges[self._cursor % len(self._judges)]
            self._cursor += 1
        return judge

    def working(self) -> bool:
        return any(judge.working() for judge in self._judges)

    def generate_inner(self, inputs, **kwargs):
        return self._next().generate_inner(inputs, **kwargs)

    def generate(self, message, **kwargs):
        return self._next().generate(message, **kwargs)

    def chat_inner(self, messages, **kwargs):
        return self._next().chat_inner(messages, **kwargs)

    def chat(self, messages, **kwargs):
        return self._next().chat(messages, **kwargs)


def _kill_stale_local_judges(ports: range | None = None) -> None:
    """Free GPU memory from leftover single-judge servers (8000/8001)."""
    import signal

    ports = ports or range(8000, 8002)
    for port in ports:
        try:
            import subprocess

            out = subprocess.check_output(
                ["pgrep", "-f", f"judge_server.*--port {port}"],
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in out.splitlines():
            pid = int(line.strip())
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    __import__("time").sleep(2)


class ShardedJudgePool:
    """Start one local judge HTTP server per GPU."""

    def __init__(
        self,
        *,
        gpu_ids: list[str],
        model_path: str,
        served_name: str = "Qwen3-8B",
        base_port: int = 8100,
    ) -> None:
        self.gpu_ids = list(gpu_ids)
        self.model_path = model_path
        self.served_name = served_name
        self.base_port = base_port
        self._procs: list[subprocess.Popen] = []
        self.base_urls: list[str] = []

    def start(self, timeout_sec: float = 900.0) -> list[str]:
        if not os.path.isfile(os.path.join(self.model_path, "config.json")):
            raise RuntimeError(
                f"Judge weights missing at {self.model_path}. "
                "Run: python -m src.utils.downloads judge"
            )

        self.base_urls = []
        self._procs = []
        for idx, gpu_id in enumerate(self.gpu_ids):
            port = self.base_port + idx
            base_url = f"http://127.0.0.1:{port}/v1"
            if judge_server_healthy(base_url):
                print(f"Judge shard gpu={gpu_id} already up at {base_url}")
                self.base_urls.append(base_url)
                continue

            cmd = [
                sys.executable,
                "-m",
                "src.evaluation.judge_server",
                "--model-path",
                self.model_path,
                "--port",
                str(port),
                "--device",
                "cuda:0",
                "--served-model-name",
                self.served_name,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_dir = LOGS_ROOT if os.path.isabs(LOGS_ROOT) else os.path.join(VTB_ROOT, LOGS_ROOT)
            os.makedirs(log_dir, exist_ok=True)
            shard_log = os.path.join(log_dir, f"judge_shard_gpu{gpu_id}_port{port}.log")
            print(f"Starting judge shard gpu={gpu_id} port={port} ... (log: {shard_log})")
            log_handle = open(shard_log, "a", encoding="utf-8")
            log_handle.write(f"\n=== judge shard gpu={gpu_id} port={port} ===\n")
            log_handle.flush()
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=VTB_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            self._procs.append(proc)
            self.base_urls.append(base_url)

        deadline = __import__("time").time() + timeout_sec
        pending = list(self.base_urls)
        wait_started = __import__("time").time()
        last_log = wait_started
        while pending and __import__("time").time() < deadline:
            now = __import__("time").time()
            if now - last_log >= 30:
                print(
                    f"Waiting for judge shards... {len(pending)} pending, "
                    f"elapsed {int(now - wait_started)}s",
                    flush=True,
                )
                last_log = now
            for url in list(pending):
                if judge_server_healthy(url):
                    pending.remove(url)
            if pending:
                __import__("time").sleep(2.0)

        if pending:
            self.stop()
            raise RuntimeError(f"Judge shards failed to become ready: {pending}")

        print(
            f"Judge pool ready: {len(self.base_urls)} shards on GPUs "
            f"[{', '.join(self.gpu_ids)}]"
        )
        return self.base_urls

    def stop(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._procs.clear()
        self.base_urls.clear()


def build_round_robin_judge(eval_cfg: dict[str, Any], base_urls: list[str]) -> RoundRobinJudge:
    from vlmeval.dataset.utils.judge_util import build_judge

    from src.evaluation.judge_config import build_judge_kwargs

    judge_kwargs = build_judge_kwargs(eval_cfg)
    model_name = str(judge_kwargs.pop("model"))
    judge_kwargs.pop("nproc", None)
    judges = []
    for base_url in base_urls:
        kwargs = dict(judge_kwargs)
        kwargs["model"] = model_name
        kwargs["api_base"] = f"{base_url.rstrip('/')}/chat/completions"
        judges.append(build_judge(**kwargs))
    return RoundRobinJudge(judges)


@contextmanager
def sharded_judge_context(
    eval_cfg: dict[str, Any] | None,
    gpu_ids: list[str] | None = None,
) -> Iterator[RoundRobinJudge | None]:
    """Context manager: start N judge servers, patch build_judge, then tear down."""
    global _POOL

    if uses_api_judge(eval_cfg):
        yield None
        return

    cfg = eval_cfg or {}
    _kill_stale_local_judges()
    _kill_stale_local_judges(range(8100, 8108))
    if gpu_ids is None:
        from src.evaluation.judge_config import parse_judge_gpu_ids

        gpus = parse_judge_gpu_ids(cfg)
    else:
        gpus = gpu_ids
    profile = resolve_judge_profile(cfg)
    pool = ShardedJudgePool(
        gpu_ids=gpus,
        model_path=str(profile["model_path"]),
        served_name=str(profile["model"]),
    )
    base_urls = pool.start()
    _POOL = pool
    round_robin = build_round_robin_judge(cfg, base_urls)

    import vlmeval.dataset.utils.judge_util as judge_util

    original_build = judge_util.build_judge
    judge_util.build_judge = lambda **_kwargs: round_robin
    try:
        yield round_robin
    finally:
        judge_util.build_judge = original_build
        pool.stop()
        _POOL = None


def ensure_sharded_judge_pool(
    eval_cfg: dict[str, Any] | None,
    gpu_ids: list[str] | None = None,
) -> list[str]:
    """Start sharded judge pool if not already running; return base URLs."""
    global _POOL
    if uses_api_judge(eval_cfg):
        return []

    if _POOL is not None and _POOL.base_urls:
        return _POOL.base_urls

    cfg = eval_cfg or {}
    if gpu_ids is None:
        from src.evaluation.judge_config import parse_judge_gpu_ids

        gpus = parse_judge_gpu_ids(cfg)
    else:
        gpus = gpu_ids
    profile = resolve_judge_profile(cfg)
    _POOL = ShardedJudgePool(
        gpu_ids=gpus,
        model_path=str(profile["model_path"]),
        served_name=str(profile["model"]),
    )
    atexit.register(_stop_pool)
    return _POOL.start()


def _stop_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.stop()
        _POOL = None


def stop_sharded_judge_pool() -> None:
    _stop_pool()
