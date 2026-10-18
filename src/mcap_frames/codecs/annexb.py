"""Annex B byte streams (H.264 / H.265): splitting into NAL units and reading header bits."""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional

PICTURE_HEAD = 64      # bytes of a picture NAL handed out by iter_nal_units: enough for any slice header field used


def iter_nal_units(data: bytes, is_picture: Optional[Callable[[int], bool]] = None) -> Iterator[bytes]:
    """NAL units without their start codes, found lazily. Trailing zeros (part of the next start code) dropped.

    When ``is_picture(first byte)`` is true, only the first PICTURE_HEAD bytes of that NAL are yielded and its
    end is not searched for, so the picture data itself is never scanned."""
    n = len(data)
    i = data.find(b"\x00\x00\x01")
    while i >= 0:
        start = i + 3
        if is_picture is not None and start < n and is_picture(data[start]):
            yield data[start:start + PICTURE_HEAD]
            j = data.find(b"\x00\x00\x01", start + PICTURE_HEAD)
            i = j
            continue
        j = data.find(b"\x00\x00\x01", start)
        end = n if j < 0 else j
        while end > start and data[end - 1] == 0:
            end -= 1
        if end > start:
            yield data[start:end]
        i = j


def split_nal_units(data: bytes) -> List[bytes]:
    return list(iter_nal_units(data))


def rbsp(nal: bytes, limit: int = 512) -> bytes:
    """Remove emulation-prevention bytes (00 00 03 → 00 00) from the first ``limit`` bytes."""
    head = nal[:limit]
    if b"\x00\x00\x03" not in head:
        return head
    out = bytearray()
    zeros = 0
    for b in head:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


class BitReader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]           # IndexError when the header is truncated
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("invalid Exp-Golomb code")
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)
