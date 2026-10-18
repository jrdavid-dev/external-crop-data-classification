"""Shared file and calendar helpers used across pipeline stages."""
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np


def write_table(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def interval_bins(year=2020):
    result = []
    for month in range(1, 13):
        first, middle = date(year, month, 1), date(year, month, 16)
        end = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
        result.extend([
            {"name": f"{year}_{month:02d}_01_15", "start": first.isoformat(), "end": middle.isoformat()},
            {"name": f"{year}_{month:02d}_16_end", "start": middle.isoformat(), "end": end.isoformat()},
        ])
    return result


def ratio(left, right):
    denominator = left + right
    return None if not np.isfinite(denominator) or abs(denominator) < 1e-10 else float((left - right) / denominator)
