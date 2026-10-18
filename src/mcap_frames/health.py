"""Health check of the image / video streams in a recording.

Every issue has a level: ``error`` (frames cannot be accessed correctly), ``warning`` (some frames are
lost or unusable, the rest are fine) or ``info`` (worth knowing, harmless for decoding).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mcap_frames import scan as scanmod
from mcap_frames.codecs.detect import VIDEO_CODECS
from mcap_frames.errors import McapFramesError
from mcap_frames.recording import Recording

ERROR, WARNING, INFO = "error", "warning", "info"
GAP_FACTOR = 1.5


@dataclass
class Issue:
    level: str
    code: str
    message: str
    frames: List[Tuple[int, int]] = field(default_factory=list)     # inclusive ranges, when relevant


@dataclass
class TopicReport:
    topic: str
    schema: str
    codec: Optional[str]
    frames: int
    duration_s: float
    fps: Optional[float]
    keyframes: int = 0
    gop: Dict[str, Optional[float]] = field(default_factory=dict)
    decodable: Dict[str, int] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)

    @property
    def level(self) -> Optional[str]:
        levels = {i.level for i in self.issues}
        return next((lv for lv in (ERROR, WARNING, INFO) if lv in levels), None)


@dataclass
class Report:
    file: str
    topics: Dict[str, TopicReport]
    bytes_read: int
    reads: int

    @property
    def ok(self) -> bool:
        return all(t.level != ERROR for t in self.topics.values())

    def to_dict(self) -> dict:
        return {"file": self.file, "ok": self.ok, "bytes_read": self.bytes_read, "reads": self.reads,
                "topics": {k: asdict(v) for k, v in self.topics.items()}}

    def summary(self) -> str:
        lines = [f"{self.file}: {'OK' if self.ok else 'PROBLEMS FOUND'}"]
        for t in self.topics.values():
            head = (f"  {t.topic}  {t.codec or '?'}  {t.frames} frames  {t.fps or '?'} fps"
                    + (f"  keyframe every {t.gop.get('most_common'):g}" if t.gop.get("most_common") else ""))
            lines.append(head)
            for i in t.issues:
                lines.append(f"    [{i.level}] {i.code}: {i.message}")
        return "\n".join(lines)


def _ranges_text(ranges: Sequence[Tuple[int, int]], limit: int = 5) -> str:
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in ranges[:limit]]
    return ", ".join(parts) + (f" (+{len(ranges) - limit} more)" if len(ranges) > limit else "")


def check_stream(stream) -> TopicReport:
    n = len(stream)
    t = stream.log_times_ns
    d = np.diff(t)
    med = float(np.median(d)) if len(d) else 0.0
    rep = TopicReport(stream.topic, stream.schema.name, None, n,
                      round((int(t[-1]) - int(t[0])) / 1e9, 3) if n > 1 else 0.0,
                      round(1e9 / med, 2) if med > 0 else None)
    if n == 0:
        rep.issues.append(Issue(ERROR, "empty", "the topic has no messages"))
        return rep
    try:
        rep.codec = stream.codec
        s = stream.scan()
    except McapFramesError as e:
        rep.issues.append(Issue(ERROR, "unreadable", str(e)))
        return rep

    keys = np.flatnonzero(s.keyframe)
    rep.keyframes = int(len(keys))
    rep.decodable = {"yes": int((s.decodable == scanmod.DECODABLE).sum()),
                     "no": int((s.decodable == scanmod.NOT_DECODABLE).sum()),
                     "unknown": int((s.decodable == scanmod.UNKNOWN).sum())}
    add = rep.issues.append

    if stream.container.recovered:
        add(Issue(WARNING, "truncated", "the file has no summary (the recording was cut off): complete chunks were "
                                        f"recovered by reading the whole file; {stream.container.ignored_tail_bytes} "
                                        "bytes of an unfinished record at the end were ignored"))
    elif not stream.container.has_message_indexes:
        add(Issue(INFO, "no-message-index", "the file has no MessageIndex records: every chunk had to be read once "
                                            "to locate frames (random access is slow until then)"))
    if (d < 0).any():
        add(Issue(ERROR, "timestamp-backwards", f"log_time goes backwards {int((d < 0).sum())} times"))
    if med > 0 and (d > GAP_FACTOR * med).any():
        gaps = np.flatnonzero(d > GAP_FACTOR * med)
        add(Issue(WARNING, "timestamp-gap",
                  f"{len(gaps)} intervals longer than {GAP_FACTOR}x the median ({med / 1e6:.1f} ms), "
                  f"longest {float(d.max()) / 1e6:.1f} ms (frames skipped by the camera or recorder)",
                  [(int(i), int(i) + 1) for i in gaps[:50]]))

    if rep.codec not in VIDEO_CODECS:
        empty = s.runs(scanmod.NOT_DECODABLE)
        if empty:
            add(Issue(WARNING, "empty-frames", f"{sum(b - a + 1 for a, b in empty)} frames are empty", empty))
        return rep

    gops = np.diff(keys)
    if len(gops):
        common = Counter(gops.tolist()).most_common(1)[0][0]
        rep.gop = {"min": int(gops.min()), "median": float(np.median(gops)), "max": int(gops.max()),
                   "most_common": float(common)}
    if not len(keys):
        add(Issue(ERROR, "no-keyframe", "the stream has no keyframe, so no frame can be decoded"))
        return rep
    if s.b_frames:
        add(Issue(ERROR, "b-frames", f"{s.b_frames} B-frames: message order differs from picture order, "
                                     "so frames cannot be addressed by number"))
    if keys[0] > 0:
        add(Issue(WARNING, "starts-mid-gop", f"the first {int(keys[0])} frames come before the first keyframe "
                                             "and cannot be decoded from this file alone", [(0, int(keys[0]) - 1)]))
    if s.params_first_seen is None:
        add(Issue(ERROR, "params-missing", "no parameter sets (SPS/PPS) anywhere in the stream"))
    else:
        missing_at_keys = int((~s.carries_params[keys]).sum())
        if missing_at_keys:
            add(Issue(INFO, "params-not-repeated",
                      f"{missing_at_keys} of {len(keys)} keyframes do not carry parameter sets; "
                      "they are taken from earlier frames"))
    after_first = s.decodable[int(keys[0]):]
    bad = [(a + int(keys[0]), b + int(keys[0])) for a, b in _runs(after_first == scanmod.NOT_DECODABLE)]
    if bad:
        add(Issue(ERROR, "missing-reference",
                  f"{sum(b - a + 1 for a, b in bad)} frames after the first keyframe depend on a missing frame "
                  f"(frames {_ranges_text(bad)}); tools that count decoded pictures will be shifted after these",
                  bad))
    if rep.gop and rep.gop["min"] != rep.gop["max"]:
        add(Issue(INFO, "irregular-gop", f"keyframe interval varies from {rep.gop['min']} to {rep.gop['max']} "
                                         "frames (often scene-cut keyframes; harmless for decoding)"))
    unknown = rep.decodable["unknown"]
    if unknown:
        why = "H.265 is not analysed yet" if rep.codec == "h265" else "the stream allows frame_num gaps"
        add(Issue(INFO, "decodability-unknown", f"for {unknown} frames it cannot be inferred whether they decode ({why})"))
    return rep


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i in np.flatnonzero(mask).tolist():
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def check(recording, topics: Optional[Sequence[str]] = None, verify_decode: bool = False) -> Report:
    """Check every image / video topic (or the given ones) of a recording or a path / URL."""
    rec = recording if isinstance(recording, Recording) else Recording(recording)
    out: Dict[str, TopicReport] = {}
    topics = list(topics or rec.video_topics)
    try:
        rec.scan(topics)                                  # one pass over the file for all topics
    except McapFramesError:
        pass                                              # reported per topic below
    for topic in topics:
        stream = rec.video(topic)
        rep = check_stream(stream)
        if verify_decode and rep.codec in VIDEO_CODECS and rep.decodable.get("no"):
            rep.issues.append(_verify(stream))
        out[topic] = rep
    return Report(rec.name, out, rec.stats.bytes, rec.stats.reads)


def _verify(stream) -> Issue:
    """Decode the whole stream in order and count pictures, to compare with the inference."""
    import av

    from mcap_frames.decode import packet
    ctx = av.CodecContext.create({"h264": "h264", "h265": "hevc"}[stream.codec], "r")
    pictures = 0
    for start in range(0, len(stream), 256):
        for p in stream.payloads(range(start, min(start + 256, len(stream)))):
            try:
                pictures += len(ctx.decode(packet(av, p.data)))
            except av.error.InvalidDataError:
                pass
    pictures += len(ctx.decode(None))
    s = stream.scan()
    return Issue(INFO, "verify-decode",
                 f"sequential decode produced {pictures} pictures for {len(stream)} messages; "
                 f"{int((s.decodable != scanmod.NOT_DECODABLE).sum())} frames were inferred decodable")
