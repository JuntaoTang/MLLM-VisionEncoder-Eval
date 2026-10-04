"""Redirect process stdout/stderr to a log file."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


@contextmanager
def redirect_output_to_file(log_path: str | None):
    """Send stdout and stderr to *log_path* for the duration of the context."""
    if not log_path:
        yield
        return

    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        # Re-wrap fd 1/2 so Python print/logging is line-buffered into the log file.
        sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
        sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
        sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(log_fd)
