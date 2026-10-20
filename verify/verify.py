"""Reproducible verification of mcap-frames, on synthetic recordings and on any directory of MCAP files.

    python verify/verify.py synthetic [--index]
    python verify/verify.py dataset DIR [--sample 300 | --all] [--index] [--json results.json]

Reference: every message of a topic is fed to a decoder in order, each packet tagged with its message number, so
every picture the decoder puts out can be traced back to the message it came from, even when the decoder drops a
frame or conceals damage. mcap-frames passes when

- every frame it says can be decoded is pixel-identical to the reference picture of that message, and
- every frame it says cannot be decoded raises FrameNotDecodable (the reference decoder either dropped it, or put
  out a concealed picture built on the wrong reference).

Also counted, for a reader that numbers frames by counting decoded pictures (the usual "decode the stream, take
the n-th picture"): frames for which it returns another message's picture, and frames for which it returns the
decoder's concealed picture (built on a missing reference) without any error. Synthetic recordings draw each picture's source frame number into the
picture; that number is checked too, and the health check is compared with what each recording should report.

--index: write an index file for each recording first and verify through it (the recording opened from the index),
so the results show that a loaded index gives what reading the recording gives.

Needs the decode extra (PyAV, Pillow). Run from a checkout of the repository: synthetic recordings are built with
tests/fixtures.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mcap_frames as mf                                   # noqa: E402
from mcap_frames.decode import decode_image, packet        # noqa: E402

AV_CODECS = {"h264": "h264", "h265": "hevc"}


def md5(rgb: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()


@dataclass
class TopicResult:
    file: str
    topic: str
    codec: str
    frames: int
    checked: int = 0
    identical: int = 0                    # decodable frames, pixel-identical to the reference
    refused: int = 0                      # frames reported undecodable (FrameNotDecodable raised)
    concealed_by_decoder: int = 0         # of those, frames the reference decoder still put out a picture for
    counting_wrong: int = 0               # frames a picture-counting reader returns for the wrong message
    counting_corrupted: int = 0           # frames it returns as the decoder's concealed picture, without error
    counting_missing: int = 0             # frame numbers past the last picture such a reader has
    number_checked: int = 0               # synthetic only: source number drawn in the picture, checked
    health: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


def reference(v: mf.VideoStream) -> Dict[int, str]:
    """Message number -> md5 of the picture a decoder puts out for it, decoding the whole topic in order."""
    pictures: Dict[int, str] = {}
    order: List[int] = []
    if v.codec in AV_CODECS:
        ctx = av.CodecContext.create(AV_CODECS[v.codec], "r")
        ctx.thread_type = "AUTO"

        def take(frames):
            for fr in frames:
                order.append(fr.pts)
                pictures[fr.pts] = md5(fr.to_ndarray(format="rgb24"))

        for start in range(0, len(v), 256):
            for i, p in zip(range(start, min(start + 256, len(v))), v._payloads(range(start, min(start + 256, len(v))))):
                pkt = packet(av, p.data)
                pkt.pts = pkt.dts = i
                try:
                    take(ctx.decode(pkt))
                except av.error.InvalidDataError:
                    pass
        take(ctx.decode(None))
    else:
        for start in range(0, len(v), 256):
            for i, p in zip(range(start, min(start + 256, len(v))), v._payloads(range(start, min(start + 256, len(v))))):
                if p.data:
                    pictures[i] = md5(decode_image(p.data))
                    order.append(i)
    reference.order = order                                # pictures in output order, for the counting reader
    return pictures


def frames_to_check(v: mf.VideoStream, decodable: Sequence[Optional[bool]], sample: Optional[int], seed: int) -> List[int]:
    n = len(v)
    if sample is None or sample >= n:
        return list(range(n))
    rng = random.Random(seed)
    picked = set(rng.sample(range(n), sample)) | {0, n - 1}
    for i in range(1, n):                                  # both sides of every change in decodability
        if decodable[i] != decodable[i - 1]:
            picked |= {i - 1, i}
    if v.codec in AV_CODECS:
        for k in v.keyframes.tolist():                     # around keyframes: last frame of a run, first of the next
            picked |= {max(0, k - 1), k}
    return sorted(picked)


def verify_topic(rec: mf.Recording, topic: str, sample: Optional[int], truth: Optional[List[int]] = None,
                 seed: int = 0) -> TopicResult:
    v = rec.video(topic)
    res = TopicResult(rec.name, topic, v.codec, len(v))
    ref = reference(v)
    order = reference.order
    decodable = [v.decodable(n) for n in range(len(v))]

    # a reader that numbers frames by counting decoded pictures
    for i in range(len(v)):
        if i not in ref:
            continue
        if i >= len(order):
            res.counting_missing += 1
        elif order[i] != i:
            res.counting_wrong += 1
        elif decodable[i] is False:
            res.counting_corrupted += 1

    wanted = frames_to_check(v, decodable, sample, seed)
    res.checked = len(wanted)
    yes = [n for n in wanted if decodable[n] is not False]
    got = {}
    for f in v.iter_frames(yes, on_error="skip"):
        got[f.index] = f
    for n in yes:
        f = got.get(n)
        if f is None:
            if decodable[n] is True:
                res.failures.append(f"frame {n}: reported decodable but not returned")
            elif n in ref:
                res.failures.append(f"frame {n}: not returned, but the reference decoder has a picture for it")
            continue
        if ref.get(n) != md5(f.image):
            res.failures.append(f"frame {n}: differs from the reference picture" if n in ref
                                else f"frame {n}: returned, but the reference decoder has no picture for it")
            continue
        res.identical += 1
        if truth is not None:
            from tests.fixtures import read_number
            res.number_checked += 1
            if read_number(f.image) != truth[n]:
                res.failures.append(f"frame {n}: shows source frame {read_number(f.image)}, expected {truth[n]}")
    for n in wanted:
        if decodable[n] is False:
            try:
                v.frame(n)
                res.failures.append(f"frame {n}: reported undecodable but a picture was returned")
            except (mf.FrameNotDecodable, mf.NoKeyframeBefore):
                res.refused += 1
                res.concealed_by_decoder += n in ref
    report = mf.check(rec, [topic]).topics[topic]
    res.health = sorted(i.code for i in report.issues)
    return res


# ---------------------------------------------------------------- synthetic recordings
P = 33_333_333


def grid(slots, noise_ms=2.0, seed=0):
    from tests import fixtures as fx
    rng = np.random.default_rng(seed)
    return [fx.T0 + s * P + int(rng.uniform(-noise_ms, noise_ms) * 1e6) for s in slots]


def synthetic_cases(work: str):
    """(name, path, topic, truth, health codes expected, health codes not expected)."""
    from tests import fixtures as fx
    units = fx.encode_h264(200, gop=10)
    cases = []

    def add(name, cams, expect=(), forbid=(), cut=None, **kw):
        path = os.path.join(work, f"case{len(cases):02d}.mcap")
        w = fx.write_recording(path, cams, **kw)
        if cut:
            data = open(path, "rb").read()
            with open(path, "wb") as f:
                f.write(data[:int(len(data) * cut)])
        cases.append((name, path, cams[0].topic, w.numbers[cams[0].topic], set(expect), set(forbid)))

    clean_forbid = {"missing-reference", "frames-skipped", "timestamp-jitter", "timing-irregular",
                    "timestamp-not-increasing", "starts-mid-gop"}
    add("clean, zstd chunks, two cameras", [fx.Camera("/cam", units), fx.Camera("/cam2", fx.encode_h264(60, gop=5))],
        forbid=clean_forbid, compression="zstd")
    add("clean, lz4 chunks, ROS 2 messages", [fx.Camera("/cam", units, kind="ros2")], forbid=clean_forbid,
        compression="lz4")
    add("clean, uncompressed chunks", [fx.Camera("/cam", units)], forbid=clean_forbid, compression="none")
    add("keyframe lost", [fx.Camera("/cam", fx.drop(units, [40]))], expect={"missing-reference"})
    add("reference frame lost", [fx.Camera("/cam", fx.drop(units, [45]))], expect={"missing-reference"})
    add("keyframe and two frames after it lost", [fx.Camera("/cam", fx.drop(units, [70, 71, 72]))],
        expect={"missing-reference"})
    add("parameter sets sent once", [fx.Camera("/cam", fx.strip_params(units))], expect={"params-not-repeated"},
        forbid={"missing-reference"})
    add("large SEI before the picture data", [fx.Camera("/cam", fx.with_big_sei(units))],
        forbid={"missing-reference"})
    add("starts in the middle of a GOP", [fx.Camera("/cam", units[5:])], expect={"starts-mid-gop"})
    add("no MessageIndex records", [fx.Camera("/cam", units)], expect={"no-message-index"}, message_index=False)
    add("messages outside chunks (fastwrite)", [fx.Camera("/cam", units)], expect={"unchunked"}, chunked=False,
        compression="none")
    add("recording cut off", [fx.Camera("/cam", units)], expect={"truncated"}, cut=0.6, chunk_size=16 * 1024)
    add("JPEG images", [fx.Camera("/cam", fx.encode_jpeg(60), kind="ros2", fmt="jpeg")], forbid=clean_forbid)
    slots = list(range(203))
    del slots[150:152], slots[60]
    add("camera skipped frames", [fx.Camera("/cam", units, stamps_ns=grid(slots[:200]))],
        expect={"frames-skipped"}, forbid={"missing-reference"})
    stamps = grid(range(200))
    stamps[30] = stamps[31]
    stamps[80] += int(0.7 * P)
    add("late and duplicated timestamps", [fx.Camera("/cam", units, stamps_ns=stamps)],
        expect={"timestamp-jitter", "timestamp-not-increasing"}, forbid={"frames-skipped"})
    bursts = [fx.T0 + int((k // 4 * 4 + (0, 1, 1.5, 1.5)[k % 4]) * P) for k in range(200)]
    add("frames delivered in bursts", [fx.Camera("/cam", units, stamps_ns=bursts)], expect={"timing-irregular"},
        forbid={"frames-skipped"})
    return cases


def open_recording(path: str, args, work: str) -> mf.Recording:
    """The recording, opened from a fresh index file when --index is given."""
    if not args.index:
        return mf.open(path)
    index = os.path.join(work, hashlib.md5(path.encode()).hexdigest() + ".frames")
    with mf.open(path) as rec:
        rec.save_index(index)
    rec = mf.open(path, index=index)
    if rec.index_status != "loaded":
        raise RuntimeError(f"{path}: index {rec.index_status}")
    return rec


def run_synthetic(args) -> List[TopicResult]:
    results = []
    with tempfile.TemporaryDirectory() as work:
        for name, path, topic, truth, expect, forbid in synthetic_cases(work):
            with open_recording(path, args, work) as rec:
                n = len(rec.video(topic))
                r = verify_topic(rec, topic, None, truth[:n])
            r.file = name
            codes = set(r.health)
            if not expect <= codes:
                r.failures.append(f"health check misses {sorted(expect - codes)}")
            if codes & forbid:
                r.failures.append(f"health check wrongly reports {sorted(codes & forbid)}")
            results.append(r)
    return results


def run_dataset(args) -> List[TopicResult]:
    paths = sorted(os.path.join(d, f) for d, _s, fs in os.walk(args.dir) for f in fs
                   if f.endswith(".mcap") and not f.startswith("._"))
    results = []
    with tempfile.TemporaryDirectory() as work:
        for i, path in enumerate(paths):
            try:
                rec = open_recording(path, args, work)
            except mf.McapFramesError as e:
                print(f"  skipped {path}: {e}", file=sys.stderr)
                continue
            with rec:
                rec.scan()
                for topic in rec.video_topics:
                    results.append(verify_topic(rec, topic, None if args.all else args.sample, seed=i))
            print(f"  {i + 1}/{len(paths)} {os.path.relpath(path, args.dir)}", file=sys.stderr)
    return results


def summarize(results: List[TopicResult], label_width: int = 44) -> bool:
    head = (f"{'recording / topic':{label_width}} {'codec':6} {'frames':>7} {'checked':>7} {'same':>7} "
            f"{'refused':>7} {'counting: wrong frame':>22} {'corrupted':>9}  result")
    print(head)
    print("-" * len(head))
    for r in results:
        label = r.file if len(results) < 40 or "/" not in r.file else f"{os.path.basename(os.path.dirname(r.file))[:30]} {r.topic}"
        wrong = f"{r.counting_wrong + r.counting_missing}" + (f" ({r.counting_missing} past the end)" if r.counting_missing else "")
        print(f"{label[:label_width]:{label_width}} {r.codec:6} {r.frames:>7} {r.checked:>7} {r.identical:>7} "
              f"{r.refused:>7} {wrong:>22} {r.counting_corrupted:>9}  {'ok' if not r.failures else 'FAILED'}")
        for f in r.failures[:5]:
            print(f"    {f}")
    total = lambda k: sum(getattr(r, k) for r in results)
    ok = not any(r.failures for r in results)
    print(f"\n{len(results)} topics, {total('frames')} frames; {total('checked')} checked: {total('identical')} "
          f"pixel-identical to the reference, {total('refused')} correctly refused as undecodable "
          f"({total('concealed_by_decoder')} of which the decoder concealed instead of dropping).")
    print(f"A reader that counts decoded pictures returns another message's picture for "
          f"{total('counting_wrong') + total('counting_missing')} frames, and a concealed (corrupted) picture without "
          f"error for {total('counting_corrupted')}.")
    print("RESULT:", "all checks passed" if ok else "FAILURES (see above)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("synthetic", help="generated recordings with known damage")
    d = sub.add_parser("dataset", help="every .mcap file under a directory")
    d.add_argument("dir")
    d.add_argument("--sample", type=int, default=300, help="random frames per topic, plus keyframes and boundaries")
    d.add_argument("--all", action="store_true", help="check every frame")
    for p in sub.choices.values():
        p.add_argument("--json", help="write the per-topic results here")
        p.add_argument("--index", action="store_true", help="verify through index files written first")
    args = ap.parse_args()
    t = time.perf_counter()
    results = run_synthetic(args) if args.mode == "synthetic" else run_dataset(args)
    ok = summarize(results)
    print(f"({time.perf_counter() - t:.0f} s, mcap-frames {mf.__version__}, PyAV {av.__version__})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
