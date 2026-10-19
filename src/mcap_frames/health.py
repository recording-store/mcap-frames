"""Health check of the image / video streams in a recording.

Every issue has a level: ``error`` (frames cannot be accessed correctly), ``warning`` (some frames are
lost or unusable, the rest are fine) or ``info`` (worth knowing, harmless for decoding).

Every issue also says what it is based on (``evidence``) and whether it is ``inferred``. Issues read from the
bitstream or the file structure are facts about the file. Issues about what happened while recording (frames
skipped by the camera, late timestamps) are inferred from timestamps, and are only as right as the timestamps.
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
GAP_FACTOR = 1.5          # an interval this many times the frame period is examined
SETTLE_FRAMES = 10        # frames on each side used to tell a missing frame from a late or early one
WHOLE_FRAME = 0.75        # a shift counts as a missing frame only past 3/4 of a period (a half is a phase slip)
PERIOD_SPAN = 30          # the frame period is first measured over this many frames
REGULAR_SHARE = 0.75      # timing is analysed only if this share of intervals is within 25% of the period


_HEADERS = "NAL unit and slice headers"
_EVIDENCE = {                                         # code -> what the issue is read from (all facts)
    "empty": "message count", "unreadable": "message data", "empty-frames": "message data",
    "unchunked": "file structure", "truncated": "file structure", "no-message-index": "file structure",
    "timestamp-backwards": "log_time",
    "missing-reference": f"{_HEADERS} (frame_num, parameter sets)", "b-frames": f"{_HEADERS} (slice type)",
    "no-keyframe": _HEADERS, "starts-mid-gop": _HEADERS, "params-missing": _HEADERS,
    "params-not-repeated": _HEADERS, "irregular-gop": _HEADERS, "decodability-unknown": _HEADERS,
    "verify-decode": "sequential decode",
}


@dataclass
class Issue:
    level: str
    code: str
    message: str
    frames: List[Tuple[int, int]] = field(default_factory=list)     # inclusive ranges, when relevant
    evidence: str = ""             # what the issue is based on
    inferred: bool = False         # a conclusion about the recording drawn from timestamps, not a fact of the file

    def __post_init__(self) -> None:
        if not self.evidence:
            self.evidence = _EVIDENCE.get(self.code, "")


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
    timing: Dict[str, object] = field(default_factory=dict)   # clock used, frame period, share of regular intervals
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
                basis = f"inferred from {i.evidence}" if i.inferred else f"from {i.evidence}"
                lines.append(f"    [{i.level}] {i.code} ({basis}): {i.message}")
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
                      round(1e9 / med, 2) if med > 0 else None)      # replaced by the capture clock when present
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

    c = stream.container
    if c.unchunked_messages:
        add(Issue(INFO, "unchunked", f"{c.unchunked_messages} messages are written outside chunks (for example ROS 2 "
                                     "'fastwrite'), with no index: the file was walked once to locate frames"))
    if stream.container.recovered:
        add(Issue(WARNING, "truncated", "the file has no summary (the recording was cut off): complete chunks were "
                                        f"recovered by reading the whole file; {stream.container.ignored_tail_bytes} "
                                        "bytes of an unfinished record at the end were ignored"))
    elif not stream.container.has_message_indexes and len(c.chunks) > (1 if c.unchunked_messages else 0):
        add(Issue(INFO, "no-message-index", "the file has no MessageIndex records: every chunk had to be read once "
                                            "to locate frames (random access is slow until then)"))
    if (d < 0).any():
        add(Issue(ERROR, "timestamp-backwards", f"log_time goes backwards {int((d < 0).sum())} times"))
    _check_timeline(rep, s.stamp_ns, t, s.reference_gaps)

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
        causes = [_explain_reference_gap(*g) for g in s.reference_gaps]
        why = "; ".join(text for text, _key in causes[:3]) + (f"; +{len(causes) - 3} more" if len(causes) > 3 else "")
        add(Issue(ERROR, "missing-reference",
                  f"{sum(b - a + 1 for a, b in bad)} frames after the first keyframe depend on a missing frame "
                  f"(frames {_ranges_text(bad)})" + (f": {why}" if why else "")
                  + "; tools that count decoded pictures will be shifted after these", bad))
    else:
        causes = []
    if rep.gop and rep.gop["min"] != rep.gop["max"]:
        lost_keys = sum(key for _text, key in causes)
        add(Issue(INFO, "irregular-gop", f"keyframe interval varies from {rep.gop['min']} to {rep.gop['max']} frames "
                  + (f"({lost_keys} keyframes were lost, see missing-reference; otherwise often scene-cut keyframes)"
                     if lost_keys else "(often scene-cut keyframes; harmless for decoding)")))
    unknown = rep.decodable["unknown"]
    if unknown:
        why = "H.265 is not analysed yet" if rep.codec == "h265" else "the stream allows frame_num gaps"
        add(Issue(INFO, "decodability-unknown", f"for {unknown} frames it cannot be inferred whether they decode ({why})"))
    return rep


def _explain_reference_gap(frame: int, expected: int, found: int, max_frame_num: int) -> Tuple[str, bool]:
    """What went missing before ``frame``, from frame_num alone (H.264).

    frame_num counts reference pictures and only goes back to 0 at an IDR. A jump forward means reference frames
    are missing; a restart without an IDR means the IDR itself is missing. Both readings are possible after a
    wrap-around, so the one needing fewer missing frames is reported."""
    jumped = (found - expected) % max_frame_num            # reference frames missing if no keyframe was lost
    if 1 <= found < jumped:                                # a keyframe (frame_num 0) and found - 1 frames after it
        after = f" and {found - 1} reference frame{'s' if found > 2 else ''}" if found > 1 else ""
        return (f"before frame {frame} a keyframe{after} went missing (frame_num restarts at {found} "
                "without a keyframe)", True)
    return (f"before frame {frame} {jumped} reference frame{'s' if jumped != 1 else ''} went missing "
            f"(frame_num {found}, expected {expected})", False)


def _timeline(t: np.ndarray, period: float) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Long intervals, split into holes and jitter.

    Against a fixed-rate grid, a missing frame shifts every later frame by one period, while a late (or early)
    frame is off the grid on its own and the frames after it are back on it. So the frames on each side of a long
    interval are compared with the grid; medians keep a few late frames from counting. Once a hole is found, later
    frames are moved back onto the grid, so holes close together are counted once (at the first of them).
    A shift of about half a period is ambiguous (a camera whose clock slips by half a frame) and is not counted:
    only shifts past WHOLE_FRAME of a period count as whole missing frames.
    Returns [(frame before the interval, frames missing)] and [frame before an interval that is only jitter]."""
    long_ = np.flatnonzero(np.diff(t) > GAP_FACTOR * period)
    off = (t - t[0]).astype(np.float64) / period - np.arange(len(t))       # in periods
    holes: List[Tuple[int, int]] = []
    jitter: List[int] = []
    for i in long_.tolist():
        before = np.median(off[max(0, i - SETTLE_FRAMES + 1):i + 1])
        missing = int(np.floor(float(np.median(off[i + 1:i + 1 + SETTLE_FRAMES]) - before) + 1 - WHOLE_FRAME))
        if missing >= 1:
            holes.append((i, missing))
            off[i + 1:] -= missing
        else:
            jitter.append(i)
    return holes, jitter


def _check_timeline(rep: TopicReport, stamps: np.ndarray, log_times: np.ndarray,
                    reference_gaps: Sequence[Tuple[int, int, int, int]]) -> None:
    """Frames missing from the timeline, timestamp jitter, and capture timestamps that do not increase.

    The capture timestamp carried in the message is used when every frame has one: it is taken closest to the
    camera, while log_time also contains delivery delays. Otherwise log_time is used."""
    add = rep.issues.append
    if len(stamps) < 2:
        return
    sd = np.diff(stamps)
    span = min(PERIOD_SPAN, len(stamps) - 1)
    use_stamps = bool((stamps >= 0).all()) and float(np.median(stamps[span:] - stamps[:-span])) > 0
    if use_stamps:
        equal, back = np.flatnonzero(sd == 0), np.flatnonzero(sd < 0)
        if len(equal) or len(back):
            pairs = sorted(np.concatenate([equal, back]).tolist())
            add(Issue(WARNING, "timestamp-not-increasing",
                      f"{len(equal)} frames carry the same capture timestamp as the frame before"
                      + (f" and {len(back)} an earlier one" if len(back) else "")
                      + "; frames cannot be told apart or ordered by capture time there",
                      [(i, i + 1) for i in pairs[:50]], "capture timestamps"))
    t, clock = (stamps, "capture") if use_stamps else (log_times, "log_time")
    # The median single interval is not the frame period when frames arrive in bursts (intervals of 0 and 2x),
    # so the period is first measured over spans of many frames. Then, counting the missing frames found, as
    # the average over the whole topic: cameras often run slightly off their nominal rate.
    period = float(np.median((t[span:] - t[:-span]) / span))
    if period <= 0:
        return
    d = np.diff(t)
    regular = float(np.mean(np.abs(d - period) <= 0.25 * period))
    clock_name = "capture timestamps" if use_stamps else "log_time"
    evidence = f"{clock_name}, {regular:.0%} of intervals within 25% of the frame period"
    rep.timing = {"clock": clock, "period_ms": round(period / 1e6, 3), "regular_share": round(regular, 4)}
    if regular < REGULAR_SHARE:
        add(Issue(INFO, "timing-irregular",
                  f"only {regular:.0%} of the {clock} intervals are close to the frame period ({period / 1e6:.1f} ms): "
                  "frames arrive in bursts, so timestamps do not tell whether frames were skipped",
                  evidence=evidence))
        return
    holes, _jitter = _timeline(t, period)
    slots = np.arange(len(t), dtype=np.float64)
    for i, m in holes:
        slots[i + 1:] += m
    period = float((t[-1] - t[0]) / slots[-1])
    if period <= 0:
        return
    holes, jitter = _timeline(t, period)
    rep.fps = round(1e9 / period, 2)
    rep.timing["period_ms"] = round(period / 1e6, 3)
    if holes:
        lost_after_encoding = {g[0] for g in reference_gaps}
        both = sum(i + 1 in lost_after_encoding for i, _m in holes)
        longest = max(float(t[i + 1] - t[i]) for i, _m in holes)
        add(Issue(WARNING, "frames-skipped",
                  f"{len(holes)} places where about {sum(m for _i, m in holes)} frames in total are missing from the "
                  f"{clock} timeline (longest interval {longest / 1e6:.1f} ms, frame period {period / 1e6:.1f} ms): "
                  "skipped by the camera, or lost before reaching the file"
                  + (f"; {both} of them where frames were lost after encoding (see missing-reference)" if both else "")
                  + ". Frame numbers are not affected, but after these places time and frame number no longer advance "
                    "together: look frames up by time",
                  [(i, i + 1) for i, _m in holes[:50]], evidence, inferred=True))
    if jitter:
        add(Issue(INFO, "timestamp-jitter",
                  f"{len(jitter)} {clock} intervals longer than {GAP_FACTOR}x the frame period are made up by "
                  "nearby frames: frames are stamped late or early, or the clock slips by part of a frame; no frame "
                  "is missing",
                  [(i, i + 1) for i in jitter[:50]], evidence, inferred=True))


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
