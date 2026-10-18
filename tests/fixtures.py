"""Synthetic recordings, including damaged ones, with the source frame number drawn into every picture.

The top band of each picture holds the frame number as black / white blocks, so after any decode the
picture can be traced back to the frame it came from, even through lossy compression.
"""

from __future__ import annotations

import io
import json
import re
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import av
import numpy as np
from mcap.writer import CompressionType, IndexType, Writer

W, H, FPS = 320, 192, 30
BITS, BAND = 10, 24
T0 = 1_760_000_000_000_000_000


def draw(i: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    y = (128 + 60 * np.sin((xx + 7 * i) / 23.0) * np.cos((yy - 3 * i) / 31.0)).astype(np.uint8)
    bw = W // BITS
    y[:BAND] = 16
    for b in range(BITS):
        if (i >> (BITS - 1 - b)) & 1:
            y[:BAND, b * bw:(b + 1) * bw] = 235
    return np.stack([y, y, y], -1)


def read_number(rgb: np.ndarray) -> int:
    luma = rgb[:BAND].mean(axis=2)
    bw = W // BITS
    bits = [int(luma[4:-4, b * bw + 6:(b + 1) * bw - 6].mean() > 128) for b in range(BITS)]
    return int("".join(map(str, bits)), 2)


@dataclass
class Unit:
    number: int          # source frame number (what the picture shows)
    data: bytes
    key: bool


def encode_h264(frames: int, gop: int = 10, bframes: int = 0) -> List[Unit]:
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = W, H, "yuv420p"
    ctx.framerate, ctx.time_base = FPS, Fraction(1, FPS)
    ctx.gop_size, ctx.max_b_frames = gop, bframes
    ctx.options = {"profile": "main" if bframes else "baseline", "preset": "veryfast",
                   "keyint_min": str(gop), "sc_threshold": "0", "crf": "28",
                   **({} if bframes else {"tune": "zerolatency"})}
    packets = []
    for i in range(frames):
        fr = av.VideoFrame.from_ndarray(draw(i), format="rgb24").reformat(format="yuv420p")
        fr.pts = i
        packets += list(ctx.encode(fr))
    packets += list(ctx.encode(None))
    # decode order; with B-frames the picture number is the presentation timestamp
    return [Unit(int(p.pts), bytes(p), bool(p.is_keyframe)) for p in packets]


def encode_jpeg(frames: int) -> List[Unit]:
    from PIL import Image
    out = []
    for i in range(frames):
        buf = io.BytesIO()
        Image.fromarray(draw(i)).save(buf, format="JPEG", quality=90)
        out.append(Unit(i, buf.getvalue(), True))
    return out


def strip_params(units: List[Unit], keep_first: bool = True) -> List[Unit]:
    """Remove SPS / PPS from every keyframe except the first: parameter sets appear only once."""
    out, seen = [], False
    for u in units:
        if u.key and (seen or not keep_first):
            parts = re.split(b"\x00\x00\x00?\x01", u.data)          # 4- and 3-byte start codes
            kept = [p for p in parts if p and (p[0] & 0x1F) not in (7, 8)]
            u = Unit(u.number, b"".join(b"\x00\x00\x00\x01" + p for p in kept), u.key)
        seen = seen or u.key
        out.append(u)
    return out


# ---------------------------------------------------------------- message encodings
def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _pb_field(num: int, payload: bytes) -> bytes:
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _pb_timestamp(stamp_ns: int) -> bytes:
    return _varint(1 << 3) + _varint(stamp_ns // 1_000_000_000) + _varint(2 << 3) + _varint(stamp_ns % 1_000_000_000)


def pb_compressed_video(stamp_ns: int, data: bytes, fmt: str, frame_id: str = "cam") -> bytes:
    return (_pb_field(1, _pb_timestamp(stamp_ns)) + _pb_field(2, frame_id.encode()) + _pb_field(3, data)
            + _pb_field(4, fmt.encode()))


def pb_compressed_image(stamp_ns: Optional[int], data: bytes, fmt: str, frame_id: str = "cam") -> bytes:
    """foxglove.CompressedImage (timestamp=1, data=2, format=3, frame_id=4); stamp None leaves it out."""
    ts = _pb_field(1, _pb_timestamp(stamp_ns)) if stamp_ns is not None else b""
    return ts + _pb_field(2, data) + _pb_field(3, fmt.encode()) + _pb_field(4, frame_id.encode())


class _CdrWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")

    def _align(self, n):
        while (len(self.buf) - 4) % n:
            self.buf.append(0)

    def i32(self, v):
        self._align(4)
        self.buf += struct.pack("<i", v)

    def u32(self, v):
        self._align(4)
        self.buf += struct.pack("<I", v)

    def string(self, s):
        b = s.encode() + b"\x00"
        self.u32(len(b))
        self.buf += b

    def seq(self, b):
        self.u32(len(b))
        self.buf += b


def cdr_compressed_image(stamp_ns: int, data: bytes, fmt: str, frame_id: str = "cam") -> bytes:
    w = _CdrWriter()
    w.i32(stamp_ns // 1_000_000_000)
    w.u32(stamp_ns % 1_000_000_000)
    w.string(frame_id)
    w.string(fmt)
    w.seq(data)
    return bytes(w.buf)


ROS2_COMPRESSED_IMAGE = b"std_msgs/Header header\nstring format\nuint8[] data\n"


# ---------------------------------------------------------------- recordings
@dataclass
class Camera:
    topic: str
    units: List[Unit]
    kind: str = "protobuf"            # "protobuf" (foxglove.CompressedVideo), "pb_image" (foxglove.CompressedImage,
                                      # no timestamp) or "ros2" (sensor_msgs CompressedImage)
    fmt: str = "h264"
    times_ns: Optional[List[int]] = None   # log_time per unit; default from the frame number


@dataclass
class Written:
    path: str
    numbers: Dict[str, List[int]] = field(default_factory=dict)   # topic → source number of message n


def write_recording(path: str, cameras: Sequence[Camera], compression: str = "zstd", chunk_size: int = 64 * 1024,
                    with_state_topic: bool = True, message_index: bool = True) -> Written:
    events = []                                      # (write position, log_time, topic, sequence, unit)
    for c in cameras:
        times = c.times_ns or [T0 + u.number * 1_000_000_000 // FPS for u in c.units]
        written = 0
        for seq, (u, t) in enumerate(zip(c.units, times)):
            written = max(written, t)                    # a camera's frames are written in order, even if
            events.append((written, t, c.topic, seq, u))  # a log_time jumps back
    if with_state_topic:
        last = max(e[1] for e in events)
        t = T0
        while t <= last:
            events.append((t, t, "/state", 0, json.dumps({"t": t}).encode()))
            t += 5_000_000
    events.sort(key=lambda e: (e[0], e[2], e[3]))
    comp = {"none": CompressionType.NONE, "zstd": CompressionType.ZSTD, "lz4": CompressionType.LZ4}[compression]
    out = Written(path)
    with open(path, "wb") as f:
        index_types = IndexType.CHUNK | IndexType.ATTACHMENT | IndexType.METADATA
        if message_index:
            index_types |= IndexType.MESSAGE
        w = Writer(f, chunk_size=chunk_size, compression=comp, index_types=index_types)
        w.start(profile="", library="mcap-frames tests")
        ids = {}
        for c in cameras:
            if c.kind == "protobuf":
                sid = w.register_schema("foxglove.CompressedVideo", "protobuf", b"")
                enc = "protobuf"
            elif c.kind == "pb_image":
                sid = w.register_schema("foxglove.CompressedImage", "protobuf", b"")
                enc = "protobuf"
            else:
                sid = w.register_schema("sensor_msgs/msg/CompressedImage", "ros2msg", ROS2_COMPRESSED_IMAGE)
                enc = "cdr"
            ids[c.topic] = (w.register_channel(c.topic, enc, sid), c)
            out.numbers[c.topic] = [u.number for u in c.units]
        state_id = None
        if with_state_topic:
            state_id = w.register_channel("/state", "json", w.register_schema("state", "jsonschema", b"{}"))
        for _w, t, topic, seq, u in events:
            if topic == "/state":
                w.add_message(state_id, log_time=t, publish_time=t, data=u, sequence=0)
                continue
            cid, c = ids[topic]
            data = (pb_compressed_video(t, u.data, c.fmt) if c.kind == "protobuf"
                    else pb_compressed_image(None, u.data, c.fmt) if c.kind == "pb_image"
                    else cdr_compressed_image(t, u.data, c.fmt))
            w.add_message(cid, log_time=t, publish_time=t, data=data, sequence=seq)
        w.finish()
    return out


def drop(units: List[Unit], numbers: Iterable[int]) -> List[Unit]:
    gone = set(numbers)
    return [u for u in units if u.number not in gone]


def sequential_pictures(units: Sequence[Unit]) -> Dict[int, bytes]:
    """Decode a complete stream in order: {source number: md5 of the RGB picture} (the reference answer)."""
    import hashlib
    ctx = av.CodecContext.create("h264", "r")
    pics = []
    for u in units:
        pics += list(ctx.decode(av.Packet(u.data)))
    pics += list(ctx.decode(None))
    out = {}
    for fr in pics:
        rgb = fr.to_ndarray(format="rgb24")
        out[read_number(rgb)] = hashlib.md5(rgb.tobytes()).hexdigest()
    return out


def with_big_sei(units: List[Unit], size: int = 8000) -> List[Unit]:
    """Put a large SEI (user data unregistered) in front of the picture data of every keyframe, as some cameras
    do: the picture data then starts well beyond the first few kilobytes of the message."""
    payload = bytes(16) + b"\x11" * (size - 16)                  # UUID + filler without emulation patterns
    n = len(payload)
    sei = b"\x06\x05" + b"\xff" * (n // 255) + bytes([n % 255]) + payload + b"\x80"
    out = []
    for u in units:
        if u.key:
            parts = [p for p in re.split(b"\x00\x00\x00?\x01", u.data) if p]
            i = next(k for k, p in enumerate(parts) if p[0] & 0x1F in (1, 5))
            parts.insert(i, sei)
            u = Unit(u.number, b"".join(b"\x00\x00\x00\x01" + p for p in parts), u.key)
        out.append(u)
    return out
