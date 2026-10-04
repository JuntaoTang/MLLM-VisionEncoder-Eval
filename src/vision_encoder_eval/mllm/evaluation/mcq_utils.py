"""MCQ helpers: expand ``options`` JSON into A/B/C/D letter choices."""

from __future__ import annotations

import ast
import json
import math
import string
from typing import Any, Mapping


def _is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    text = str(val).strip()
    return not text or text.lower() == "nan"


def parse_options_field(raw: Any) -> list[str]:
    """Parse LMUData ``options`` cell (list / JSON / python-literal) into choice texts."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]

    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []

    if isinstance(parsed, (list, tuple)):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return []


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    # pandas Series / named indexing
    try:
        if hasattr(row, "index") and key not in row.index:
            return None
        return row[key]
    except Exception:
        return None


def mcq_choice_map(row: Any) -> dict[str, str]:
    """Return ``{A: text, B: text, ...}`` from letter columns or ``options`` list.

    Local MMMU TSVs store choices only in ``options`` (no A–D columns). VTB and
    VLMEvalKit prompts need lettered choices for letter-answer MCQ.
    """
    choices: dict[str, str] = {}
    for letter in string.ascii_uppercase:
        val = _row_get(row, letter)
        if _is_empty(val):
            continue
        choices[letter] = str(val).strip()

    if choices:
        return choices

    opts = parse_options_field(_row_get(row, "options"))
    for i, opt in enumerate(opts):
        if i >= len(string.ascii_uppercase):
            break
        choices[string.ascii_uppercase[i]] = opt
    return choices


def expand_options_columns_on_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> list[str]:
    """Mutate rows to add A/B/C/... from ``options``; return updated fieldnames."""
    if "options" not in fieldnames:
        return fieldnames
    if any(letter in fieldnames for letter in string.ascii_uppercase):
        return fieldnames

    max_n = 0
    for row in rows:
        opts = parse_options_field(row.get("options"))
        for i, opt in enumerate(opts):
            row[string.ascii_uppercase[i]] = opt
        max_n = max(max_n, len(opts))

    if max_n <= 0:
        return fieldnames

    out = list(fieldnames)
    insert_at = out.index("options") + 1
    for i in range(max_n):
        letter = string.ascii_uppercase[i]
        if letter not in out:
            out.insert(insert_at + i, letter)
    return out


def ensure_mcq_letter_columns_xlsx(xlsx_path: str) -> str:
    """Expand ``options`` → A/B/C/... in-place for VLMEvalKit MCQ judge scoring."""
    import os

    import pandas as pd

    if not os.path.isfile(xlsx_path):
        return xlsx_path
    df = pd.read_excel(xlsx_path)
    if "options" not in df.columns:
        return xlsx_path

    changed = False
    for idx, row in df.iterrows():
        choice_map = mcq_choice_map(row)
        for letter, text in choice_map.items():
            if letter not in df.columns:
                df[letter] = None
                changed = True
            if _is_empty(df.at[idx, letter]):
                df.at[idx, letter] = text
                changed = True

    if not changed:
        return xlsx_path

    out = xlsx_path.replace(".xlsx", "_with_abcd.xlsx")
    df.to_excel(out, index=False)
    return out
