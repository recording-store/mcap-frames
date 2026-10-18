"""Which codec a stream uses: the message's format field first, the bytes only when it is empty."""

from __future__ import annotations

from typing import Optional

from mcap_frames.codecs.annexb import split_nal_units

IMAGE_CODECS = ("jpeg", "png", "webp")
VIDEO_CODECS = ("h264", "h265")


def codec_from_format(fmt: str) -> Optional[str]:
    f = (fmt or "").lower()
    for key, codec in (("264", "h264"), ("avc", "h264"), ("265", "h265"), ("hevc", "h265"),
                       ("jpeg", "jpeg"), ("jpg", "jpeg"), ("png", "png"), ("webp", "webp")):
        if key in f:
            return codec
    return None


def sniff_codec(data: bytes) -> Optional[str]:
    """Guess from the bytes. Video is only recognised once a parameter set appears; otherwise None.

    (A plain H.264 P-slice header byte also reads as an H.265 parameter set, so a first frame without
    parameter sets cannot tell the two apart.)"""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for nal in split_nal_units(data[:4096]):
        if nal[0] & 0x1F in (7, 8):
            return "h264"
        if len(nal) > 1 and (nal[0] >> 1) & 0x3F in (32, 33, 34) and nal[1] & 7:
            return "h265"
    return None
