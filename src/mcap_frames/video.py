"""One image / video stream: frame n is the n-th message of the channel.

Getting frame n reads backwards from n to the nearest keyframe (and, if that keyframe does not carry its
parameter sets, further back to where they last appeared), then decodes forward. The frames in between
are checked: if a reference picture is missing, FrameNotDecodable is raised rather than returning a
picture built on the wrong reference or a neighbouring frame.
"""

from __future__ import annotations

import os
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from mcap.records import Channel, Schema

from mcap_frames import scan as scanmod
from mcap_frames.codecs import h264, h265
from mcap_frames.codecs.detect import IMAGE_CODECS, VIDEO_CODECS, codec_from_format, sniff_codec
from mcap_frames.container import Container
from mcap_frames.errors import FrameNotDecodable, NoKeyframeBefore, UnsupportedStream
from mcap_frames.index import FrameIndex, build_index
from mcap_frames.messages import Extractor, Payload

_PAYLOAD_CACHE = 1024
_SCAN_BATCH = 2048
_SCAN_HEAD = 4096          # bytes read per frame when scanning headers (uncompressed chunks)


@dataclass
class Frame:
    index: int
    log_time_ns: int
    stamp_ns: Optional[int]
    image: np.ndarray                  # RGB, (height, width, 3), uint8
    decoded_from: int                  # the keyframe decoding started at


@dataclass
class EncodedRun:
    """Access units from a keyframe to a target frame, ready for any decoder (parameter sets included)."""

    codec: str
    first_index: int
    units: List[bytes]


@dataclass
class _Located:
    keyframe: int
    prefix: bytes
    state: Optional[bool]              # True: references complete; None: cannot tell (decoder decides)


class VideoStream:
    """Frames of one image / video topic. Not thread-safe: give each thread its own ``mf.open(...)``."""

    def __init__(self, container: Container, channel: Channel, schema: Schema, extract: Extractor):
        self._container = container
        self.channel = channel
        self.schema = schema
        self.topic = channel.topic
        self._extract = extract
        self._index: Optional[FrameIndex] = None
        self._codec: Optional[str] = None
        self._scan: Optional[scanmod.ScanResult] = None
        self._stamp_order: Optional[np.ndarray] = None
        self._cache: "OrderedDict[int, Payload]" = OrderedDict()

    # ------------------------------------------------------------------ basics
    @property
    def _frame_index(self) -> FrameIndex:
        if self._index is None:
            self._index = build_index(self._container, self.channel.id)
        return self._index

    def __len__(self) -> int:
        return len(self._frame_index)

    @property
    def log_times_ns(self) -> np.ndarray:
        """log_time of every frame, in frame order (read from the index, no scan needed)."""
        return self._frame_index.log_time_ns

    @property
    def stamps_ns(self) -> np.ndarray:
        """Capture timestamp carried in each message, in frame order; -1 where a message has none.

        Needs the headers of every frame (one scan, see Recording.scan)."""
        return self._scanned().stamp_ns

    @property
    def keyframes(self) -> np.ndarray:
        """Frame numbers of the keyframes (needs one scan, see Recording.scan)."""
        return np.flatnonzero(self._scanned().keyframe)

    @property
    def codec(self) -> str:
        if self._codec is None:
            if not len(self):
                raise UnsupportedStream(f"{self.topic} has no messages")
            for i in range(min(len(self), 64)):
                p = self._payloads([i])[0]
                codec = codec_from_format(p.format) or sniff_codec(p.data)
                if codec:
                    self._codec = codec
                    break
            else:
                raise UnsupportedStream(f"cannot tell the codec of {self.topic} from its first frames")
        return self._codec

    def _payloads(self, frames: Sequence[int]) -> List[Payload]:
        frames = list(frames)
        got: Dict[int, Payload] = {n: self._cache[n] for n in frames if n in self._cache}
        missing = sorted({n for n in frames if n not in got})
        if missing:
            raw = self._container.read_messages([self._frame_index.refs[n] for n in missing])
            got.update((n, self._extract(data)) for n, data in zip(missing, raw))
        for n in frames:                                   # refresh the cache only after the answer is complete
            self._cache[n] = got[n]
            self._cache.move_to_end(n)
        while len(self._cache) > _PAYLOAD_CACHE:
            self._cache.popitem(last=False)
        return [got[n] for n in frames]

    def index_at(self, time_ns: int, mode: str = "nearest", clock: str = "log_time") -> int:
        """Frame number for a time: the nearest frame, the last one at or before, or the first at or after.

        ``clock`` is "log_time" (from the index) or "capture" (the timestamp carried in each message: closer to
        when the picture was taken, but it needs one scan, and every frame must carry one). Frames with equal
        times: "after" and "nearest" give the first of them, "before" the last."""
        if clock == "log_time":
            times, order = self.log_times_ns, self._frame_index.time_order
        elif clock == "capture":
            times = self.stamps_ns
            if (times < 0).any():
                raise UnsupportedStream(f"{self.topic}: not every frame carries a capture timestamp")
            if self._stamp_order is None:
                self._stamp_order = np.argsort(times, kind="stable")
            order = self._stamp_order
        else:
            raise ValueError("clock must be 'log_time' or 'capture'")
        t = times[order]
        if not len(t):
            raise IndexError(f"{self.topic} has no frames")
        if mode == "after":
            i = int(np.searchsorted(t, time_ns, side="left"))
            if i >= len(t):
                raise IndexError(f"no frame of {self.topic} at or after {time_ns}")
            return int(order[i])
        if mode == "before":
            j = int(np.searchsorted(t, time_ns, side="right")) - 1
            if j < 0:
                raise IndexError(f"no frame of {self.topic} at or before {time_ns}")
            return int(order[j])
        if mode != "nearest":
            raise ValueError("mode must be 'nearest', 'before' or 'after'")
        i = int(np.searchsorted(t, time_ns, side="left"))
        cands = [k for k in (i - 1, i) if 0 <= k < len(t)]
        best = min(cands, key=lambda k: abs(int(t[k]) - time_ns))
        best = int(np.searchsorted(t, t[best], side="left"))      # the first of equal times
        return int(order[best])

    # ------------------------------------------------------------------ whole-stream scan
    def _scanned(self) -> scanmod.ScanResult:
        """Every frame's headers, read once (cached). To scan several topics of one file, Recording.scan() reads
        each chunk only once."""
        if self._scan is None:
            scan_streams([self])
        return self._scan

    def decodable(self, n: int) -> Optional[bool]:
        """True / False, or None when it cannot be inferred (H.265, or H.264 streams that allow frame_num gaps).

        Worked out from frame_num without decoding; needs one scan (see Recording.scan)."""
        d = int(self._scanned().decodable[self._check(n)])
        return None if d == scanmod.UNKNOWN else bool(d)

    # ------------------------------------------------------------------ locating the run for frame n
    def _check(self, n: int) -> int:
        if not 0 <= n < len(self):
            raise IndexError(f"frame {n} is outside {self.topic} (0..{len(self) - 1})")
        return n

    def _parser(self):
        if self.codec == "h264":
            return h264.H264State(), h264.parse_unit, h264.params_available
        if self.codec == "h265":
            return h265.H265State(), h265.parse_unit, h265.params_available
        raise UnsupportedStream(f"{self.topic}: {self.codec!r} is not a video codec")

    def _locate(self, n: int) -> _Located:
        """Keyframe to start from, parameter-set prefix for it, and whether frame n's references are complete."""
        if self._scan is not None:
            return self._locate_with_scan(n)
        state, parse, _avail = self._parser()
        # 1. walk back to the nearest keyframe
        k, window = None, 16
        hi = n
        while k is None:
            lo = max(0, hi - window + 1)
            units = [parse(p.data, state.copy()) if p.data else None for p in self._payloads(range(lo, hi + 1))]
            for i in range(hi, lo - 1, -1):
                u = units[i - lo]
                if u is not None and u.keyframe:
                    k = i
                    break
            else:
                if lo == 0:
                    raise NoKeyframeBefore(f"{self.topic}: no keyframe at or before frame {n}")
                hi, window = lo - 1, min(window * 2, 1024)
        # 2. parameter sets: carried by the keyframe, or the last ones before it
        p_at = k
        while True:
            u = parse(self._payloads([p_at])[0].data, state.copy())
            has = (u.has_sps and u.has_pps) if self.codec == "h264" else (u.has_vps and u.has_sps and u.has_pps)
            if has:
                break
            if p_at == 0:
                raise FrameNotDecodable(f"{self.topic}: no parameter sets at or before keyframe {k}")
            p_at -= 1
        return self._track(k, p_at, n)

    def _locate_with_scan(self, n: int) -> _Located:
        s = self._scan
        if s.decodable[n] == scanmod.NOT_DECODABLE:
            if not s.keyframe[: n + 1].any():
                raise NoKeyframeBefore(f"{self.topic}: no keyframe at or before frame {n}")
            raise FrameNotDecodable(f"{self.topic}: frame {n} depends on a missing frame")
        keys = np.flatnonzero(s.keyframe[: n + 1])
        k = int(keys[-1])
        carriers = np.flatnonzero(s.carries_params[: k + 1])
        if not len(carriers):
            raise FrameNotDecodable(f"{self.topic}: no parameter sets at or before keyframe {k}")
        return self._track(k, int(carriers[-1]), n)

    def _track(self, k: int, params_at: int, n: int) -> _Located:
        state, parse, _avail = self._parser()
        if params_at < k:
            parse(self._payloads([params_at])[0].data, state)
        prefix_needed = False
        result: Optional[bool] = True
        if self.codec == "h264":
            tracker = h264.FrameNumTracker()
            for i, p in zip(range(k, n + 1), self._payloads(range(k, n + 1))):
                if not p.data:
                    raise FrameNotDecodable(f"{self.topic}: frame {i} is empty")
                u = h264.parse_unit(p.data, state)
                if u.slice_type == h264.SLICE_B:
                    raise UnsupportedStream(f"{self.topic} has B-frames: frame numbers do not match picture order")
                if i == k:
                    prefix_needed = not (u.has_sps and u.has_pps)
                    pps_id = u.pps_id
                    result = tracker.start(u)
                else:
                    result = tracker.step(u)
                if result is False:
                    raise FrameNotDecodable(f"{self.topic}: frame {n} depends on a frame missing before frame {i}")
            prefix = state.param_prefix(pps_id) if prefix_needed else b""
        else:
            u = h265.parse_unit(self._payloads([k])[0].data, state)
            prefix = b"" if (u.has_vps and u.has_sps and u.has_pps) else state.param_prefix()
            result = None if n > k else True
        return _Located(k, prefix, result)

    # ------------------------------------------------------------------ access
    def encoded(self, n: int) -> EncodedRun:
        """Raw access units from the nearest keyframe to frame n (no decoding)."""
        self._check(n)
        if self.codec in IMAGE_CODECS:
            return EncodedRun(self.codec, n, [self._payloads([n])[0].data])
        loc = self._locate(n)
        units = [p.data for p in self._payloads(range(loc.keyframe, n + 1))]
        units[0] = loc.prefix + units[0]
        return EncodedRun(self.codec, loc.keyframe, units)

    def frame(self, n: int) -> Frame:
        return self.frames([n])[0]

    def frames(self, frames: Sequence[int], workers: Optional[int] = None, decoder_threads: Optional[int] = None,
               on_error: str = "raise") -> List[Frame]:
        """Decode several frames, returned in request order (all held in memory: for many frames, use
        iter_frames). See iter_frames for the arguments."""
        return list(self.iter_frames(frames, workers, decoder_threads, on_error))

    def iter_frames(self, frames: Sequence[int], workers: Optional[int] = None,
                    decoder_threads: Optional[int] = None, on_error: str = "raise") -> Iterator[Frame]:
        """Decode frames and yield them in request order, as soon as they are ready.

        Frames sharing a keyframe are decoded in one pass. Runs that start at different keyframes are decoded on
        ``workers`` threads at once (default: one per CPU core), a few runs ahead of what has been yielded, so
        memory stays at a few runs' worth of pictures as long as frames of one run are requested together (for
        example in ascending order). ``decoder_threads`` is passed to each decoder (0 = FFmpeg decides); by
        default each decoder gets one thread when several runs are decoded at once, and FFmpeg's choice when
        there is only one run, so both cases use all cores.

        ``on_error``: "raise" (default) raises FrameNotDecodable / NoKeyframeBefore for a frame that cannot be
        decoded (frames whose headers show it are reported before anything is yielded; a decoder failure is
        reported when its run is reached); "skip" leaves such frames out (``Frame.index`` tells which frames came
        back). Frame numbers outside the stream always raise IndexError."""
        from mcap_frames.decode import decode_image, decode_run

        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        order = [self._check(n) for n in frames]
        if not order:
            return
        first_pos: Dict[int, int] = {}
        for pos, n in enumerate(order):
            first_pos.setdefault(n, pos)
        failed: set = set()
        skippable = (FrameNotDecodable, NoKeyframeBefore)

        # 1. plan: one job per run (video) or per frame (images), in the order they are first asked for
        is_image = self.codec in IMAGE_CODECS
        if not is_image and self.codec not in VIDEO_CODECS:
            raise UnsupportedStream(f"{self.topic}: cannot decode {self.codec!r}")
        groups: Dict[int, Tuple[Optional[_Located], List[int]]] = {}
        for n in sorted(first_pos):
            if is_image:
                groups[n] = (None, [n])
                continue
            try:
                loc = self._locate(n)
            except skippable:
                if on_error == "raise":
                    raise
                failed.add(n)
                continue
            groups.setdefault(loc.keyframe, (loc, []))[1].append(n)
        jobs = sorted(groups.values(), key=lambda g: min(first_pos[t] for t in g[1]))

        n_workers = max(1, min(len(jobs), workers or os.cpu_count() or 1))
        threads = decoder_threads if decoder_threads is not None else (1 if n_workers > 1 else 0)

        def prepare(job):                                  # reads on this thread (the cache is not thread-safe)
            loc, targets = job
            if is_image:
                p = self._payloads(targets)[0]
                return targets[0], targets, [p.data], {targets[0]: p.stamp_ns}
            payloads = self._payloads(range(loc.keyframe, targets[-1] + 1))
            units = [p.data for p in payloads]
            units[0] = loc.prefix + units[0]
            return loc.keyframe, targets, units, {t: payloads[t - loc.keyframe].stamp_ns for t in targets}

        def run(prepared):
            k, targets, units, stamps = prepared
            try:
                if is_image:
                    if not units[0]:
                        raise FrameNotDecodable(f"{self.topic}: frame {k} is empty")
                    return k, targets, {k: decode_image(units[0])}, stamps, None
                return k, targets, decode_run(units, self.codec, [t - k for t in targets], threads), stamps, None
            except skippable as e:
                return k, targets, None, stamps, e

        # 2. decode a few jobs ahead; yield in request order as frames become ready
        left = Counter(order)
        ready: Dict[int, Tuple[np.ndarray, int, Optional[int]]] = {}
        pos = 0

        def take(result) -> None:
            k, targets, got, stamps, error = result
            if error is not None:
                if on_error == "raise":
                    raise error
                failed.update(targets)
                return
            for t in targets:
                ready[t] = (got[t - k] if not is_image else got[k], k, stamps[t])

        def drain() -> Iterator[Frame]:
            nonlocal pos
            while pos < len(order):
                n = order[pos]
                if n in failed:
                    pos += 1
                    continue
                if n not in ready:
                    return
                image, start, stamp = ready[n]
                left[n] -= 1
                if not left[n]:
                    del ready[n]
                pos += 1
                yield Frame(n, int(self.log_times_ns[n]), stamp, image, start)

        if n_workers == 1:
            for job in jobs:
                take(run(prepare(job)))
                yield from drain()
            return
        with ThreadPoolExecutor(n_workers, thread_name_prefix="mcap-frames-decode") as pool:
            pending: "deque" = deque()
            todo = iter(jobs)
            for job in todo:
                pending.append(pool.submit(run, prepare(job)))
                if len(pending) >= n_workers:
                    break
            while pending:
                take(pending.popleft().result())
                nxt = next(todo, None)
                if nxt is not None:
                    pending.append(pool.submit(run, prepare(nxt)))
                yield from drain()


def scan_streams(streams: Sequence["VideoStream"]) -> None:
    """Scan several streams of one container in a single pass over the file.

    Messages of all streams are visited in file order, so every chunk is read (and decompressed) once, no
    matter how many cameras share it; only the head of each message is kept."""
    todo = [v for v in streams if v._scan is None]
    if not todo:
        return
    container = todo[0]._container
    entries = [(v, i, r) for v in todo for i, r in enumerate(v._frame_index.refs)]
    entries.sort(key=lambda e: (e[2].chunk, e[2].offset))
    heads: Dict[int, List[Optional[Payload]]] = {id(v): [None] * len(v) for v in todo}
    for start in range(0, len(entries), _SCAN_BATCH):
        batch = entries[start:start + _SCAN_BATCH]
        for (v, i, _r), (data, truncated) in zip(batch, container.read_message_heads([e[2] for e in batch], _SCAN_HEAD)):
            heads[id(v)][i] = v._extract(data, truncated)
    for v in todo:
        v._scan = scanmod.scan_payloads(v.codec, heads[id(v)], len(v), refetch=lambda i, v=v: v._payloads([i])[0])
