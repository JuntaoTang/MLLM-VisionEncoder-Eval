"""Lightweight OpenAI-compatible judge API for Qwen3-8B (MMMU/MMBench MCQ extraction)."""

from __future__ import annotations

from vision_encoder_eval.core.runtime import asset_path

import argparse
import json
import os
import re
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_MODEL = None
_TOKENIZER = None

_THINKING_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
_MCQ_LETTER_RE = re.compile(r"^[A-GZ]$", re.IGNORECASE)
_MCQ_LABEL_RE = re.compile(
    r"(?:your output|answer|output|prediction)\s*[:：]\s*([A-GZ])\b",
    re.IGNORECASE,
)


def _strip_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text).strip()


def extract_mcq_letter(raw: str) -> str:
    """Reduce verbose judge output to a single MCQ letter for VLMEvalKit can_infer()."""
    text = _strip_thinking(str(raw or "")).strip()
    if not text:
        return text

    match = _MCQ_LABEL_RE.search(text)
    if match:
        return match.group(1).upper()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if _MCQ_LETTER_RE.fullmatch(line):
            return line.upper()

    tokens = text.split()
    if len(tokens) == 1 and _MCQ_LETTER_RE.fullmatch(tokens[0]):
        return tokens[0].upper()
    if tokens and _MCQ_LETTER_RE.fullmatch(tokens[0]) and len(tokens) <= 3:
        return tokens[0].upper()

    return text


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "\n".join(
                str(item.get("text", item)) for item in content if isinstance(item, dict)
            )
        else:
            text = str(content)
        normalized.append({"role": role, "content": text})
    return normalized


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    assert _TOKENIZER is not None
    normalized = _normalize_messages(messages)
    try:
        return _TOKENIZER.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return _TOKENIZER.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )


def _visible_gpu_max_memory_gib(reserve_gib: float = 1.5) -> dict[int, str] | None:
    """Per-visible-GPU free memory for device_map=auto (accounts for other processes)."""
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: nvidia-smi free-memory probe failed: {exc}", flush=True)
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not str(visible).strip():
        phys_ids = list(range(len(out.strip().splitlines())))
    else:
        phys_ids = [int(p.strip()) for p in str(visible).split(",") if p.strip()]

    free_mib = [int(line.strip()) for line in out.strip().splitlines() if line.strip()]
    max_memory: dict[int, str] = {}
    for local_idx, phys in enumerate(phys_ids):
        if phys < 0 or phys >= len(free_mib):
            continue
        usable = max(0.0, free_mib[phys] / 1024.0 - reserve_gib)
        max_memory[local_idx] = f"{usable:.1f}GiB"
    if not max_memory:
        return None
    print(f"  max_memory from free VRAM: {max_memory}", flush=True)
    return max_memory


def _load_model(model_path: str, device: str) -> None:
    global _MODEL, _TOKENIZER
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading judge model from {model_path} on {device} ...", flush=True)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is not available for local judge server.")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    n_visible = torch.cuda.device_count() if device.startswith("cuda") else 0
    max_memory = _visible_gpu_max_memory_gib() if n_visible >= 1 else None
    # Multiple visible GPUs (CUDA_VISIBLE_DEVICES) → shard with device_map=auto.
    if n_visible > 1:
        print(f"  device_map=auto across {n_visible} visible GPUs", flush=True)
        kwargs: dict[str, Any] = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map="auto",
        )
        if max_memory:
            kwargs["max_memory"] = max_memory
        _MODEL = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        # Single visible GPU: still respect free VRAM when another job is resident.
        if max_memory and 0 in max_memory:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            _MODEL = _MODEL.to(device)
    _MODEL.eval()
    print("Judge model ready.", flush=True)


def _generate(messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
    import torch

    assert _MODEL is not None and _TOKENIZER is not None
    prompt = _messages_to_prompt(messages)
    inputs = _TOKENIZER(prompt, return_tensors="pt")
    try:
        target = _MODEL.device
    except Exception:
        target = next(_MODEL.parameters()).device
    inputs = {k: v.to(target) for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_tokens)),
        "do_sample": float(temperature) > 0,
        "pad_token_id": _TOKENIZER.eos_token_id,
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = float(temperature)
    with torch.inference_mode():
        output = _MODEL.generate(**inputs, **gen_kwargs)
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    raw = _TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_mcq_letter(raw)


class JudgeHandler(BaseHTTPRequestHandler):
    server_version = "VTBJudge/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/v1/models", "/v1/models/"):
            payload = {
                "object": "list",
                "data": [{"id": os.environ.get("JUDGE_SERVED_NAME", "Qwen3-8B"), "object": "model"}],
            }
            self._json(200, payload)
            return
        if self.path in ("/health", "/healthz"):
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        messages = req.get("messages") or []
        max_tokens = int(req.get("max_tokens", 32))
        temperature = float(req.get("temperature", 0))
        answer = _generate(messages, max_tokens=max_tokens, temperature=temperature)
        model_name = req.get("model") or os.environ.get("JUDGE_SERVED_NAME", "Qwen3-8B")
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._json(200, payload)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    from vision_encoder_eval.mllm.utils.attention import disable_flash_attention, ensure_cuda_nvcc_shim

    disable_flash_attention()
    ensure_cuda_nvcc_shim()

    parser = argparse.ArgumentParser(description="Start local Qwen3-8B judge API")
    parser.add_argument("--model-path", default=asset_path('runtime', 'model/judge'))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--served-model-name", default="Qwen3-8B")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.model_path, "config.json")):
        print(f"Missing judge weights: {args.model_path}", file=sys.stderr)
        return 1

    os.environ["JUDGE_SERVED_NAME"] = args.served_model_name
    _load_model(args.model_path, args.device)
    httpd = ThreadingHTTPServer((args.host, args.port), JudgeHandler)
    print(
        f"Judge API listening on http://{args.host}:{args.port}/v1 "
        f"(model={args.served_model_name})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down judge server.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
