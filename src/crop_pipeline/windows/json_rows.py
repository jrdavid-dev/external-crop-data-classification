"""Read a JSON object with large row arrays one record at a time."""

import json
import re

_CHUNK_SIZE = 256 * 1024
_MAX_VALUE_SIZE = 8 * 1024 * 1024
_WHITESPACE = re.compile(r"[ \t\r\n]*")


class _Reader:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.index = 0
        self.offset = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self):
        if self.eof:
            return False
        chunk = self.stream.read(_CHUNK_SIZE)
        self.eof = not chunk
        if not chunk:
            return False
        self.buffer = self.buffer[self.index:] + chunk
        self.index = 0
        return bool(chunk)

    def _advance(self, end):
        consumed = self.buffer[self.index:end]
        self.offset += len(consumed) if consumed.isascii() else len(consumed.encode("utf-8"))
        self.index = end

    def peek(self):
        while True:
            self._advance(_WHITESPACE.match(self.buffer, self.index).end())
            if self.index < len(self.buffer):
                return self.buffer[self.index]
            if not self._fill():
                return ""

    def expect(self, character):
        if self.peek() != character:
            raise ValueError(f"Expected {character!r} at byte {self.offset}")
        self._advance(self.index + 1)

    def value(self):
        self.peek()
        start = self.offset
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.index)
            except json.JSONDecodeError:
                if len(self.buffer) - self.index >= _MAX_VALUE_SIZE or not self._fill():
                    raise
            else:
                # A scalar can end at a chunk boundary with more digits to come.
                if end == len(self.buffer) and self._fill():
                    continue
                self._advance(end)
                return value, (start, self.offset - start)


def _members(reader, arrays, container=None):
    reader.expect("{")
    while reader.peek() != "}":
        key, _ = reader.value()
        reader.expect(":")
        if container is not None and key == container:
            yield from _members(reader, arrays)
        elif key in arrays:
            reader.expect("[")
            while reader.peek() != "]":
                value, span = reader.value()
                yield key, value, span
                if reader.peek() == "]":
                    break
                reader.expect(",")
                if reader.peek() == "]":
                    raise ValueError(f"Trailing comma at byte {reader.offset}")
            reader.expect("]")
        else:
            value, _ = reader.value()
            yield key, value, None
        if reader.peek() == "}":
            break
        reader.expect(",")
        if reader.peek() == "}":
            raise ValueError(f"Trailing comma at byte {reader.offset}")
    reader.expect("}")


def read_rows(path, arrays, *, container=None):
    """Yield (key, value, byte span) for array items and file metadata."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = _Reader(stream)
        yield from _members(reader, arrays, container)
        if reader.peek():
            raise ValueError("Unexpected content after handoff")
