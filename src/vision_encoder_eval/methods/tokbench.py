from statistics import fmean
TEXT_BUCKETS = ((0.02, 0.03), (0.03, 0.04), (0.04, 1.0))
FACE_BUCKETS = ((0.1, 0.2), (0.2, 0.3), (0.3, 1.0))

def bucket_means(rows: list[dict], key: str, buckets) -> list[float]:
    values = []
    for lower, upper in buckets:
        selected = [float(row[key]) for row in rows if lower <= float(row["ratio"]) < upper]
        if not selected:
            raise RuntimeError(f"empty {key} bucket [{lower}, {upper})")
        values.append(fmean(selected))
    return values
