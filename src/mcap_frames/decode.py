"""Decoding to RGB arrays (optional dependency: ``pip install mcap-frames[decode]``)."""

from __future__ import annotations

import io
from typing import Dict, Iterable, List, Sequence

import numpy as np

from mcap_frames.errors import FrameNotDecodable, UnsupportedStream

_AV_CODECS = {"h264": "h264", "h265": "hevc"}


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as e:                                        # pragma: no cover
        raise ImportError(f"decoding needs {module}: pip install 'mcap-frames[decode]'") from e


def packet(av, data):
    """A packet FFmpeg can safely read.

    FFmpeg's bitstream readers may read up to AV_INPUT_BUFFER_PADDING_SIZE (64) bytes past the end of the input,
    so input buffers must be padded. ``av.Packet(bytes)`` (PyAV 18) wraps the Python buffer as is, without
    padding: when a frame happens to end right before an unmapped page, decoding crashes the process. A packet
    allocated with ``av.Packet(size)`` is padded; the data is copied into it."""
    pkt = av.Packet(len(data))
    if len(data):
        memoryview(pkt)[:] = data
    return pkt


def decode_run(units: Sequence[bytes], codec: str, want: Iterable[int], threads: int = 1) -> Dict[int, np.ndarray]:
    """Decode a run of access units that starts at a keyframe; return {position: RGB array} for ``want``.

    Every unit must produce exactly one picture. If the decoder drops one, positions would no longer
    match message numbers, so this raises instead of guessing."""
    if codec not in _AV_CODECS:
        raise UnsupportedStream(f"cannot decode {codec!r} as video")
    av = _require("av")
    want = set(want)
    ctx = av.CodecContext.create(_AV_CODECS[codec], "r")
    if threads != 1:                                   # 0 = let FFmpeg pick; frame threading keeps one picture per unit
        ctx.thread_type = "AUTO"
        ctx.thread_count = threads
    pictures: List = []
    out: Dict[int, np.ndarray] = {}

    def take(frames) -> None:
        for fr in frames:
            pos = len(pictures)
            pictures.append(None)
            if pos in want:
                out[pos] = fr.to_ndarray(format="rgb24")

    try:
        for unit in units:
            take(ctx.decode(packet(av, unit)))
        take(ctx.decode(None))
    except av.error.InvalidDataError as e:
        raise FrameNotDecodable(f"decoder rejected the stream: {e}") from e
    if len(pictures) != len(units):
        raise FrameNotDecodable(f"decoder produced {len(pictures)} pictures for {len(units)} access units")
    return out


def decode_image(data: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:                                        # pragma: no cover
        raise ImportError("decoding images needs Pillow: pip install 'mcap-frames[decode]'") from e
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))
