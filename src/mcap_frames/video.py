"""One image / video stream: frame n is the n-th message of the channel.

Getting frame n reads backwards from n to the nearest keyframe (and, if that keyframe does not carry its
parameter sets, further back to where they last appeared), then decodes forward. The frames in between
are checked: if a reference picture is missing, FrameNotDecodable is raised rather than returning a
picture built on the wrong reference or a neighbouring frame.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    def __init__(self, container: Container, channel: Channel, schema: Schema, extract: Extractor):
        self.container = container
        self.channel = channel
        self.schema = schema
        self.topic = channel.topic
        self._extract = extract
        self._index: Optional[FrameIndex] = None
        self._codec: Optional[str] = None
        self._scan: Optional[scanmod.ScanResult] = None
        self._cache: "OrderedDict[int, Payload]" = OrderedDict()

    # ------------------------------------------------------------------ basics
    @property
    def index(self) -> FrameIndex:
        if self._index is None:
            self._index = build_index(self.container, self.channel.id)
        return self._index

    def __len__(self) -> int:
        return len(self.index)

    @property
    def log_times_ns(self) -> np.ndarray:
        return self.index.log_time_ns

    @property
    def codec(self) -> str:
        if self._codec is None:
            if not len(self):
                raise UnsupportedStream(f"{self.topic} has no messages")
            for i in range(min(len(self), 64)):
                p = self.payloads([i])[0]
                codec = codec_from_format(p.format) or sniff_codec(p.data)
                if codec:
                    self._codec = codec
                    break
            else:
                raise UnsupportedStream(f"cannot tell the codec of {self.topic} from its first frames")
        return self._codec

    def payloads(self, frames: Sequence[int]) -> List[Payload]:
        frames = list(frames)
        got: Dict[int, Payload] = {n: self._cache[n] for n in frames if n in self._cache}
        missing = sorted({n for n in frames if n not in got})
        if missing:
            raw = self.container.read_messages([self.index.refs[n] for n in missing])
            got.update((n, self._extract(data)) for n, data in zip(missing, raw))
        for n in frames:                                   # refresh the cache only after the answer is complete
            self._cache[n] = got[n]
            self._cache.move_to_end(n)
        while len(self._cache) > _PAYLOAD_CACHE:
            self._cache.popitem(last=False)
        return [got[n] for n in frames]

    def index_at(self, time_ns: int, mode: str = "nearest") -> int:
        """Frame number for a log_time: the nearest frame, the last one at or before, or the first at or after."""
        order = self.index.time_order
        t = self.log_times_ns[order]
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
        return int(order[min(cands, key=lambda k: abs(int(t[k]) - time_ns))])

    # ------------------------------------------------------------------ whole-stream scan
    def scan(self) -> scanmod.ScanResult:
        """Read every frame's headers once (cached). Needed for decodable(), health checks and exports.

        To scan several topics of one file, use Recording.scan(), which reads each chunk only once."""
        if self._scan is None:
            scan_streams([self])
        return self._scan

    def decodable(self, n: int) -> Optional[bool]:
        """True / False, or None when it cannot be inferred (H.265, or H.264 streams that allow frame_num gaps)."""
        d = int(self.scan().decodable[self._check(n)])
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
            units = [parse(p.data, state.copy()) if p.data else None for p in self.payloads(range(lo, hi + 1))]
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
            u = parse(self.payloads([p_at])[0].data, state.copy())
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
            parse(self.payloads([params_at])[0].data, state)
        prefix_needed = False
        result: Optional[bool] = True
        if self.codec == "h264":
            tracker = h264.FrameNumTracker()
            for i, p in zip(range(k, n + 1), self.payloads(range(k, n + 1))):
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
            u = h265.parse_unit(self.payloads([k])[0].data, state)
            prefix = b"" if (u.has_vps and u.has_sps and u.has_pps) else state.param_prefix()
            result = None if n > k else True
        return _Located(k, prefix, result)

    # ------------------------------------------------------------------ access
    def encoded(self, n: int) -> EncodedRun:
        """Raw access units from the nearest keyframe to frame n (no decoding)."""
        self._check(n)
        if self.codec in IMAGE_CODECS:
            return EncodedRun(self.codec, n, [self.payloads([n])[0].data])
        loc = self._locate(n)
        units = [p.data for p in self.payloads(range(loc.keyframe, n + 1))]
        units[0] = loc.prefix + units[0]
        return EncodedRun(self.codec, loc.keyframe, units)

    def frame(self, n: int) -> Frame:
        return self.frames([n])[0]

    def frames(self, frames: Sequence[int], workers: Optional[int] = None,
               decoder_threads: Optional[int] = None) -> List[Frame]:
        """Decode several frames; frames sharing a keyframe are decoded in one pass. Returned in request order.

        Runs that start at different keyframes are independent and are decoded on ``workers`` threads at once
        (default: one per CPU core). ``decoder_threads`` is passed to each decoder (0 = FFmpeg decides); by
        default each decoder gets one thread when several runs are decoded at once, and FFmpeg's choice when
        there is only one run (a single frame), so both cases use all cores."""
        from mcap_frames.decode import decode_image, decode_run

        wanted = sorted({self._check(n) for n in frames})
        images: Dict[int, Tuple[np.ndarray, int]] = {}
        if self.codec in IMAGE_CODECS:
            for n, p in zip(wanted, self.payloads(wanted)):
                if not p.data:
                    raise FrameNotDecodable(f"{self.topic}: frame {n} is empty")
                images[n] = (decode_image(p.data), n)
        elif self.codec in VIDEO_CODECS:
            groups: List[Tuple[_Located, List[int]]] = []
            for n in wanted:
                loc = self._locate(n)
                if groups and groups[-1][0].keyframe == loc.keyframe:
                    groups[-1][1].append(n)
                else:
                    groups.append((loc, [n]))
            jobs = []
            for loc, targets in groups:                    # read on this thread (cache is not thread-safe)
                units = [p.data for p in self.payloads(range(loc.keyframe, targets[-1] + 1))]
                units[0] = loc.prefix + units[0]
                jobs.append((loc.keyframe, targets, units))

            n_workers = min(len(jobs), workers or os.cpu_count() or 1)
            threads = decoder_threads if decoder_threads is not None else (1 if n_workers > 1 else 0)

            def run(job):
                k, targets, units = job
                return k, targets, decode_run(units, self.codec, [t - k for t in targets], threads)

            if n_workers > 1:
                with ThreadPoolExecutor(n_workers, thread_name_prefix="mcap-frames-decode") as pool:
                    results = list(pool.map(run, jobs))
            else:
                results = [run(j) for j in jobs]
            for k, targets, got in results:
                for t in targets:
                    images[t] = (got[t - k], k)
        else:
            raise UnsupportedStream(f"{self.topic}: cannot decode {self.codec!r}")
        out = []
        for n in frames:
            image, start = images[n]
            stamp = self.payloads([n])[0].stamp_ns
            out.append(Frame(n, int(self.log_times_ns[n]), stamp, image, start))
        return out


def scan_streams(streams: Sequence["VideoStream"]) -> None:
    """Scan several streams of one container in a single pass over the file.

    Messages of all streams are visited in file order, so every chunk is read (and decompressed) once, no
    matter how many cameras share it; only the head of each message is kept."""
    todo = [v for v in streams if v._scan is None]
    if not todo:
        return
    container = todo[0].container
    entries = [(v, i, r) for v in todo for i, r in enumerate(v.index.refs)]
    entries.sort(key=lambda e: (e[2].chunk, e[2].offset))
    heads: Dict[int, List[Optional[Payload]]] = {id(v): [None] * len(v) for v in todo}
    for start in range(0, len(entries), _SCAN_BATCH):
        batch = entries[start:start + _SCAN_BATCH]
        for (v, i, _r), (data, truncated) in zip(batch, container.read_message_heads([e[2] for e in batch], _SCAN_HEAD)):
            heads[id(v)][i] = v._extract(data, truncated)
    for v in todo:
        v._scan = scanmod.scan_payloads(v.codec, heads[id(v)], len(v), refetch=lambda i, v=v: v.payloads([i])[0])
