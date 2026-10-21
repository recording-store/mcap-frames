"""Writing frames out: images with a manifest, or an MP4 (stream copy, no re-encoding)."""

from __future__ import annotations

import io
import json
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence

from mcap_frames import scan as scanmod
from mcap_frames.codecs.detect import VIDEO_CODECS
from mcap_frames.decode import packet
from mcap_frames.errors import FrameNotDecodable, NoKeyframeBefore, UnsupportedStream

STATUS_OK, STATUS_NOT_DECODABLE, STATUS_NO_KEYFRAME = "ok", "not_decodable", "no_keyframe_before"


def export_frames(stream, frames: Sequence[int], out_dir, fmt: str = "jpeg", quality: int = 90) -> List[dict]:
    """Write the given frames as images; frames that cannot be decoded are listed with their status.

    Writes ``<out_dir>/<frame:08d>.<ext>`` and ``<out_dir>/frames.json`` (one row per requested frame). Images are
    written as they are decoded, so memory does not grow with the number of frames."""
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = {"jpeg": "jpg", "png": "png"}[fmt]
    kwargs = {"quality": quality} if fmt == "jpeg" else {}
    written = {}
    for f in stream.iter_frames(frames, on_error="skip"):
        if f.index not in written:
            name = f"{f.index:08d}.{ext}"
            Image.fromarray(f.image).save(out / name, **kwargs)
            written[f.index] = (name, f.stamp_ns)
    rows = []
    for n in frames:
        row = {"frame": n, "log_time_ns": int(stream.log_times_ns[n])}
        if n in written:
            row.update(status=STATUS_OK, file=written[n][0], stamp_ns=written[n][1])
        else:
            row.update(status=_why_not_decodable(stream, n), file=None)
        rows.append(row)
    (out / "frames.json").write_text(json.dumps({"topic": stream.topic, "frames": rows}, indent=1), encoding="utf-8")
    return rows


def _why_not_decodable(stream, n: int) -> str:
    """Status of a frame iter_frames left out (only the frames that failed are looked at again)."""
    if stream.codec not in VIDEO_CODECS:
        return STATUS_NOT_DECODABLE                # an image that could not be read (empty or damaged)
    try:
        stream._locate(n)
    except NoKeyframeBefore:
        return STATUS_NO_KEYFRAME
    except FrameNotDecodable:
        return STATUS_NOT_DECODABLE
    return STATUS_NOT_DECODABLE                    # located, but the decoder could not produce it


def to_mp4(stream, output, start_index: Optional[int] = None, end_index: Optional[int] = None,
           allow_undecodable: bool = False) -> dict:
    """Copy frames [start, end] (inclusive) into an MP4 without re-encoding. The start moves back to a keyframe.

    Times in the MP4 are log_time relative to the first written frame. By default this refuses ranges that
    contain frames depending on a missing frame, because players would show broken pictures there."""
    import av

    if stream.codec not in VIDEO_CODECS:
        raise UnsupportedStream(f"{stream.topic}: {stream.codec} frames cannot be put in an MP4 without re-encoding")
    n = len(stream)
    end = n - 1 if end_index is None else min(end_index, n - 1)
    start = 0 if start_index is None else start_index
    if not 0 <= start <= end:
        raise ValueError(f"{stream.topic}: no frames between {start_index} and {end_index} (0..{n - 1})")
    if stream._scan is not None:                     # a scan is at hand: use it
        s, offset = stream._scan, 0
        keys = [k for k in range(start, -1, -1) if s.keyframe[k] and s.decodable[k] == scanmod.DECODABLE]
        if not keys:
            raise NoKeyframeBefore(f"{stream.topic}: no usable keyframe at or before frame {start}")
        start = keys[0]
    else:                                            # read only the range, from the keyframe before it
        start, params_at = stream._keyframe_before(start)
        s, offset = stream._scan_range(start, params_at, end), start
    if s.b_frames:
        raise UnsupportedStream(f"{stream.topic} has B-frames")
    bad = [i for i in range(start, end + 1) if s.decodable[i - offset] == scanmod.NOT_DECODABLE]
    if bad and not allow_undecodable:
        raise FrameNotDecodable(f"{stream.topic}: frames {bad[0]}..{bad[-1]} ({len(bad)} frames) in the range "
                                "depend on a missing frame; pass allow_undecodable=True to write them anyway")
    first = stream.encoded(start)                        # keyframe with its parameter sets
    tb = Fraction(1, 1_000_000)
    template = av.open(io.BytesIO(first.units[0]), format={"h264": "h264", "h265": "hevc"}[stream.codec])
    container = av.open(output, "w", format="mp4")
    t0 = int(stream.log_times_ns[start])
    last = -1
    try:
        vs = container.add_stream_from_template(template.streams.video[0])
        vs.time_base = tb
        for i0 in range(start, end + 1, 256):
            idx = list(range(i0, min(i0 + 256, end + 1)))
            for i, p in zip(idx, stream._payloads(idx)):
                pkt = packet(av, first.units[0] if i == start else p.data)
                pkt.stream = vs
                pkt.time_base = tb
                pts = max((int(stream.log_times_ns[i]) - t0) // 1000, last + 1)
                pkt.pts = pkt.dts = pts
                nxt = int(stream.log_times_ns[i + 1]) if i + 1 < n else None
                pkt.duration = max(1, (nxt - int(stream.log_times_ns[i])) // 1000) if nxt else 1
                pkt.is_keyframe = bool(s.keyframe[i - offset])
                container.mux(pkt)
                last = pts
    finally:
        container.close()
        template.close()
    return {"first_frame": start, "last_frame": end, "frames": end - start + 1, "undecodable_frames": len(bad)}
