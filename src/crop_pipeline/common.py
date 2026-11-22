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


class _JsonStream:
    """Incremental reader for one top-level JSON object, holding one value at a time."""

    _whitespace = " \t\r\n"

    def __init__(self, stream, chunk_size):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.position, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()

    def _fill(self):
        chunk = self.stream.read(self.chunk_size)
        self.eof = not chunk
        self.buffer = self.buffer[self.position:] + chunk
        self.position = 0
        return bool(chunk)

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position] in self._whitespace:
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if not self._fill():
                raise ValueError("Unexpected end of JSON file")

    def expect(self, character):
        if self.peek() != character:
            raise ValueError(f"Expected {character!r} in JSON file")
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof or not self._fill():
                    raise
                continue
            # A number at the end of the buffer may continue in the next chunk.
            if end == len(self.buffer) and not self.eof and self._fill():
                continue
            self.position = end
            return value


def iter_json_object(path, arrays=(), chunk_size=1 << 24):
    """Yield ``(key, value)`` for a top-level JSON object without loading it whole.

    Arrays named in ``arrays`` are yielded element by element as ``(key, element)``;
    every other top-level value is yielded whole. Elements of other arrays are
    decoded as part of their value, so only name small values there.
    """
    arrays = set(arrays)
    with Path(path).open(encoding="utf-8") as stream:
        reader = _JsonStream(stream, chunk_size)
        reader.expect("{")
        if reader.peek() == "}":
            return
        while True:
            key = reader.value()
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            reader.expect(":")
            if key in arrays and reader.peek() == "[":
                reader.expect("[")
                if reader.peek() != "]":
                    while True:
                        yield key, reader.value()
                        if reader.peek() == "]":
                            break
                        reader.expect(",")
                reader.expect("]")
            else:
                yield key, reader.value()
            if reader.peek() == "}":
                return
            reader.expect(",")


def iter_json_array(path, key, large=("fixed", "aligned")):
    """Yield one top-level array's elements; other ``large`` arrays are skipped element-wise."""
    for name, value in iter_json_object(path, arrays={key, *large}):
        if name == key:
            yield value


def json_scalars(path, large):
    """Return the top-level values other than the named large arrays."""
    return {name: value for name, value in iter_json_object(path, arrays=large) if name not in large}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Large row collections can exhaust disk and memory when pretty-printed.
    rows = value.values() if isinstance(value, dict) else (value,)
    if any(isinstance(item, list) and len(item) >= 50_000 for item in rows):
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)
    else:
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
