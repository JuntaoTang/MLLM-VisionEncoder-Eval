import ast
import csv
import json
import os
import re
from typing import Any, Optional


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"')


def _parse_answer(raw: Any) -> tuple[list[str], str]:
    """Return (all_answers, primary_gt) from an LMUData TSV answer field."""
    if raw is None:
        return [], ""

    text = _strip_quotes(str(raw))
    if not text:
        return [], ""

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text

    if isinstance(parsed, list):
        answers = [str(x) for x in parsed if str(x).strip()]
        return answers, answers[0] if answers else ""

    return [str(parsed)], str(parsed)


def parse_image_path_field(raw: Any) -> list[str]:
    """Parse LMUData image_path cells (scalar path or stringified list)."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]

    text = _strip_quotes(str(raw))
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if parsed is not None:
            return [str(parsed).strip()]

    return [text]


def serialize_image_path_field(paths: list[str], *, force_list: bool = False) -> str:
    """Serialize paths back to the LMUData / VLMEvalKit TSV cell format."""
    if force_list or len(paths) != 1:
        return str(list(paths))
    return paths[0]


def resolve_image_path(image_path: str, image_dir: str) -> str:
    path = _strip_quotes(str(image_path))
    # Caller may pass an unparsed list cell; take the first real path.
    if path.startswith("[") and path.endswith("]"):
        parsed = parse_image_path_field(path)
        if not parsed:
            return os.path.join(image_dir, path)
        if len(parsed) == 1:
            path = parsed[0]
        else:
            # Multi-image: resolve first for scalar APIs; prefer parse_image_path_field.
            path = parsed[0]

    if path and os.path.isfile(path):
        return path

    # Local TSVs often store repo-relative paths like images/test/MME/OCR/0001.jpg
    lmu_root = os.environ.get("LMUData") or os.path.dirname(os.path.dirname(image_dir))
    data_root = os.path.dirname(lmu_root) if os.path.basename(lmu_root) in (".lmudata", "lmudata") else lmu_root
    candidates: list[str] = []
    norm = path.replace("\\", "/").lstrip("./")
    if norm.startswith("images/"):
        candidates.append(os.path.join(lmu_root, norm))
        # Also try /cache/data/images/test/... when LMUData is /cache/data/.lmudata
        parent = os.path.dirname(lmu_root)
        candidates.append(os.path.join(parent, norm))
        # Drop the leading images/<split>/ when image_dir already points at dataset folder
        parts = norm.split("/")
        if len(parts) >= 3 and parts[0] == "images":
            # images/test/MME/OCR/0001.jpg -> OCR/0001.jpg under image_dir=.../MME
            rest_after_dataset = "/".join(parts[3:]) if len(parts) > 3 else parts[-1]
            if rest_after_dataset:
                candidates.append(os.path.join(image_dir, rest_after_dataset))
            candidates.append(os.path.join(image_dir, parts[-1]))

    basename = os.path.basename(path)
    # Bail out of corrupted cells like "['1_1.jpg']" mistaken for a filename.
    if "[" in basename or "]" in basename:
        recovered = parse_image_path_field(basename)
        if recovered:
            basename = os.path.basename(recovered[0])

    candidates.append(os.path.join(image_dir, basename))
    # Preserve subdirectory under image_dir when present in the original path.
    if "/" in norm:
        # Prefer trailing path under the dataset image root (e.g. OCR/0001.jpg).
        for i, part in enumerate(norm.split("/")):
            if part and part == os.path.basename(image_dir.rstrip("/")):
                rel = "/".join(norm.split("/")[i + 1 :])
                if rel:
                    candidates.append(os.path.join(image_dir, rel))
                break

    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand

    return candidates[-1] if candidates else os.path.join(image_dir, basename)


def resolve_image_paths(image_path: Any, image_dir: str) -> list[str]:
    """Resolve every entry in an image_path cell (keeps multi-image lists)."""
    return [resolve_image_path(p, image_dir) for p in parse_image_path_field(image_path)]


def load_lmu_tsv(
    tsv_path: str,
    image_dir: str,
    max_samples: Optional[int] = None,
) -> list[dict]:
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"Eval TSV not found: {tsv_path}")

    samples: list[dict] = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            question = _strip_quotes(row.get("question", ""))
            if not question:
                continue

            image_path = row.get("image_path") or row.get("image") or ""
            resolved_image = resolve_image_path(image_path, image_dir)
            answers, primary_gt = _parse_answer(row.get("answer"))
            gt = _strip_quotes(row.get("multiple_choice_answer", "")) or primary_gt

            samples.append(
                {
                    "index": _strip_quotes(row.get("index", str(len(samples)))),
                    "question": question,
                    "answer": gt,
                    "all_answers": answers,
                    "image": resolved_image,
                    "image_name": os.path.basename(resolved_image),
                }
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples


_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_RE = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA_RE = re.compile(r"(\d)(,)(\d)")


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or _COMMA_RE.search(text):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_RE.sub("", out)


def normalize_vqa_answer(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\t", " ").strip()
    text = _process_punctuation(text)
    text = text.strip("'").strip('"').strip(")").strip("(")
    return text.strip().lower()


def vqa_score(prediction: str, all_answers: list[str]) -> float:
    """Official-style VQA soft accuracy averaged over annotator references."""
    if not all_answers:
        return 0.0

    pred = normalize_vqa_answer(prediction)
    gt = [normalize_vqa_answer(x) for x in all_answers]
    scores = []
    for i in range(len(gt)):
        others = [item for j, item in enumerate(gt) if j != i]
        matching = sum(1 for item in others if item == pred)
        scores.append(min(1.0, matching / 3.0))
    return sum(scores) / len(scores)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_vqa_answer(prediction) == normalize_vqa_answer(ground_truth)
