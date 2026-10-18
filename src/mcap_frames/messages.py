"""Getting the image bytes out of a message, without generated message classes.

Only the few fields needed here are parsed (data, format, timestamp), straight from the wire format:

==================================  ==========  ==================================================
schema                              encoding    layout
==================================  ==========  ==================================================
foxglove.CompressedVideo            protobuf    timestamp=1, frame_id=2, data=3, format=4
foxglove.CompressedImage            protobuf    timestamp=1, data=2, format=3, frame_id=4
foxglove_msgs/msg/CompressedVideo   cdr         Time timestamp, string frame_id, uint8[] data, string format
foxglove_msgs/msg/CompressedImage   cdr         Time timestamp, string frame_id, uint8[] data, string format
sensor_msgs/msg/CompressedImage     cdr         Header header (Time stamp, string frame_id), string format, uint8[] data
==================================  ==========  ==================================================
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Optional

from mcap.records import Channel, Schema


@dataclass
class Payload:
    data: bytes
    format: str
    stamp_ns: Optional[int]
    truncated: bool = False          # only the head of the message was read: ``data`` is its first bytes


Extractor = Callable[..., Payload]  # (raw message bytes, truncated=False) -> Payload


# ---------------------------------------------------------------------------- protobuf
def _varint(buf: bytes, pos: int):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _protobuf_fields(buf: bytes):
    """Yield (field number, wire type, value); length-delimited values are (start, end) slices."""
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = (pos, pos + length)
            pos += length
        elif wire == 1:
            value = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8
        elif wire == 5:
            value = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _protobuf_timestamp(buf: bytes) -> int:
    seconds = nanos = 0
    for field, wire, value in _protobuf_fields(buf):
        if field == 1 and wire == 0:
            seconds = value - (1 << 64) if value >= 1 << 63 else value
        elif field == 2 and wire == 0:
            nanos = value
    return seconds * 1_000_000_000 + nanos


def _protobuf_extractor(data_field: int, format_field: int) -> Extractor:
    def extract(raw: bytes, truncated: bool = False) -> Payload:
        data, fmt, stamp = b"", "", None
        try:
            for field, wire, value in _protobuf_fields(raw):
                if wire != 2:
                    continue
                start, end = value
                if field == data_field:
                    data = raw[start:end]                 # shorter than declared when truncated
                elif field == format_field:
                    fmt = raw[start:end].decode("utf-8", "replace")
                elif field == 1:
                    stamp = _protobuf_timestamp(raw[start:end])
        except (IndexError, struct.error):
            if not truncated:
                raise
        return Payload(data, fmt, stamp, truncated)
    return extract


# ---------------------------------------------------------------------------- ROS 2 CDR
class _Cdr:
    def __init__(self, raw: bytes):
        if len(raw) < 4:
            raise ValueError("CDR message shorter than its encapsulation header")
        kind = raw[1]
        if kind not in (0, 1):
            raise ValueError(f"unsupported CDR encapsulation {kind:#x}")
        self.endian = "<" if kind == 1 else ">"
        self.buf, self.pos = raw, 4

    def _align(self, n: int) -> None:
        self.pos += (-(self.pos - 4)) % n

    def u32(self) -> int:
        self._align(4)
        v = struct.unpack_from(self.endian + "I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def i32(self) -> int:
        self._align(4)
        v = struct.unpack_from(self.endian + "i", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def time_ns(self) -> int:
        sec = self.i32()
        return sec * 1_000_000_000 + self.u32()

    def string(self) -> str:
        n = self.u32()
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s.rstrip(b"\x00").decode("utf-8", "replace")

    def bytes_seq(self) -> bytes:
        n = self.u32()
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s


def _cdr_foxglove(raw: bytes, truncated: bool = False) -> Payload:
    r = _Cdr(raw)
    stamp = r.time_ns()
    r.string()                                      # frame_id
    data = r.bytes_seq()
    try:
        fmt = r.string()                            # after the data: missing when only the head was read
    except struct.error:
        if not truncated:
            raise
        fmt = ""
    return Payload(data, fmt, stamp, truncated)


def _cdr_sensor_compressed_image(raw: bytes, truncated: bool = False) -> Payload:
    r = _Cdr(raw)
    stamp = r.time_ns()
    r.string()                                      # header.frame_id
    fmt = r.string()
    return Payload(r.bytes_seq(), fmt, stamp, truncated)


_EXTRACTORS = {
    ("foxglove.CompressedVideo", "protobuf"): _protobuf_extractor(3, 4),
    ("foxglove.CompressedImage", "protobuf"): _protobuf_extractor(2, 3),
    ("foxglove_msgs/msg/CompressedVideo", "cdr"): _cdr_foxglove,
    ("foxglove_msgs/msg/CompressedImage", "cdr"): _cdr_foxglove,
    ("sensor_msgs/msg/CompressedImage", "cdr"): _cdr_sensor_compressed_image,
}

SUPPORTED_SCHEMAS = sorted({name for name, _enc in _EXTRACTORS})


def extractor_for(schema: Optional[Schema], channel: Channel) -> Optional[Extractor]:
    if schema is None:
        return None
    return _EXTRACTORS.get((schema.name, channel.message_encoding))
