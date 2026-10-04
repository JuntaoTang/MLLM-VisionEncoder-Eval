"""Attention-backend helpers for local inference / judge servers."""

from __future__ import annotations

import os


def disable_flash_attention() -> None:
    """Prefer SDPA/eager over flash-attn (often missing or ABI-mismatched)."""
    os.environ.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "sdpa")
    os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_CHECK", "1")
    # Some stacks also read this flag.
    os.environ["DISABLE_FLASH_ATTN"] = "1"


def ensure_cuda_nvcc_shim() -> str | None:
    """DeepSpeed/transformers may probe ``nvcc`` at import time.

    On machines without a CUDA toolkit install, create a tiny shim so model load
    can proceed with prebuilt torch wheels (no compilation needed).
    """
    import shutil
    import stat
    import tempfile

    if shutil.which("nvcc"):
        return None
    root = os.path.join(tempfile.gettempdir(), "vtb_cuda_shim")
    bindir = os.path.join(root, "bin")
    os.makedirs(bindir, exist_ok=True)
    nvcc = os.path.join(bindir, "nvcc")
    if not os.path.isfile(nvcc):
        with open(nvcc, "w", encoding="utf-8") as f:
            f.write(
                "#!/bin/sh\n"
                'echo "nvcc: NVIDIA (R) Cuda compiler driver"\n'
                'echo "Cuda compilation tools, release 12.4, V12.4.99"\n'
            )
        os.chmod(nvcc, os.stat(nvcc).st_mode | stat.S_IEXEC)
    os.environ["CUDA_HOME"] = root
    os.environ["CUDA_PATH"] = root
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    return root
