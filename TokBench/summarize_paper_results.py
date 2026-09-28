#!/usr/bin/env python3
"""Aggregate native-256 TokBench scores used by the paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


TEXT_FILES = (
    "ic13.json",
    "ic15.json",
    "tt.json",
    "textocr.json",
    "sroie.json",
    "cord.json",
    "infograph.json",
    "docvqa.json",
)
TEXT_BUCKETS = ((0.02, 0.03), (0.03, 0.04), (0.04, 1.0))
FACE_BUCKETS = ((0.1, 0.2), (0.2, 0.3), (0.3, 1.0))
CONFIGS = (
    ("TokLIP-S", "toklip_s"),
    ("TokLIP-L", "toklip_l"),
    ("UniTok", "unitok"),
    ("VILA-U", "vilau_7b_256"),
)


def bucket_means(rows: list[dict], key: str, buckets) -> list[float]:
    values = []
    for lower, upper in buckets:
        selected = [float(row[key]) for row in rows if lower <= float(row["ratio"]) < upper]
        if not selected:
            raise RuntimeError(f"empty {key} bucket [{lower}, {upper})")
        values.append(fmean(selected))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent / "image_outputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "output")
    args = parser.parse_args()

    text_payloads = {
        name: json.loads((args.input_dir / name).read_text(encoding="utf-8"))
        for name in TEXT_FILES
    }
    face_payload = json.loads((args.input_dir / "face.json").read_text(encoding="utf-8"))
    summary_rows = []
    detail_rows = []
    for display_name, key in CONFIGS:
        text_rows = []
        for filename, payload in text_payloads.items():
            try:
                details = payload[key]["256"]["details"]
            except KeyError as error:
                raise KeyError(f"missing {key}/256 in {filename}") from error
            for image in details.values():
                text_rows.extend(image["results"])
        face_rows = []
        try:
            face_details = face_payload[key]["256"]["details"]
        except KeyError as error:
            raise KeyError(f"missing {key}/256 in face.json") from error
        for image in face_details.values():
            face_rows.extend(image["results"])

        acc = bucket_means(text_rows, "accuracy", TEXT_BUCKETS)
        ned = bucket_means(text_rows, "ned", TEXT_BUCKETS)
        face = bucket_means(face_rows, "similarity", FACE_BUCKETS)
        summary_rows.append(
            {
                "vision_encoder": display_name,
                "reconstruction_tokenizer": "TokLIP shared VQ" if key.startswith("toklip") else display_name,
                "resolution": 256,
                "t_acc_percent": 100.0 * fmean(acc),
                "t_ned_percent": 100.0 * fmean(ned),
                "f_sim": fmean(face),
                "status": "evaluated",
            }
        )
        for metric, buckets, values in (
            ("T-ACC", TEXT_BUCKETS, acc),
            ("T-NED", TEXT_BUCKETS, ned),
            ("F-Sim", FACE_BUCKETS, face),
        ):
            for (lower, upper), value in zip(buckets, values):
                detail_rows.append(
                    {
                        "vision_encoder": display_name,
                        "metric": metric,
                        "ratio_min": lower,
                        "ratio_max": upper,
                        "value": value,
                    }
                )

    summary_rows.append(
        {
            "vision_encoder": "UniAR-BSQ",
            "reconstruction_tokenizer": "",
            "resolution": 256,
            "t_acc_percent": "",
            "t_ned_percent": "",
            "f_sim": "",
            "status": "not applicable: no released reconstruction decoder",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "tokbench_discrete_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    details_path = args.output_dir / "tokbench_bucket_details.csv"
    with details_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)
    print(summary_path)
    print(details_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
