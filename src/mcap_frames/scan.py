"""Reading every frame's headers once: keyframes, parameter sets, B-frames, and which frames can be decoded."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from mcap_frames.codecs import h264, h265
from mcap_frames.messages import Payload

DECODABLE, NOT_DECODABLE, UNKNOWN = 1, 0, -1


@dataclass
class ScanResult:
    codec: str
    keyframe: np.ndarray              # bool: decoding can start at this frame
    carries_params: np.ndarray        # bool: the frame carries the parameter sets it needs
    decodable: np.ndarray             # int8: DECODABLE / NOT_DECODABLE / UNKNOWN
    b_frames: int
    stamp_ns: np.ndarray              # int64, -1 where the message has no timestamp of its own
    gaps_allowed: Optional[bool]      # H.264 gaps_in_frame_num_allowed_flag
    params_first_seen: Optional[int]  # first frame that carried parameter sets
    reference_gaps: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # H.264: (frame, expected frame_num, frame_num found, max_frame_num) where references go missing
    params: Optional[np.ndarray] = None   # int32 per frame: at keyframes, the parameter sets decoding from there
    param_sets: List[bytes] = field(default_factory=list)   # needs (row in param_sets); -1 elsewhere or unknown

    def __len__(self) -> int:
        return len(self.keyframe)

    def runs(self, value: int) -> List[Tuple[int, int]]:
        """Inclusive (first, last) ranges of frames whose ``decodable`` equals ``value``."""
        idx = np.flatnonzero(self.decodable == value)
        out: List[Tuple[int, int]] = []
        for i in idx.tolist():
            if out and i == out[-1][1] + 1:
                out[-1] = (out[-1][0], i)
            else:
                out.append((i, i))
        return out


def scan_payloads(codec: str, payloads: Iterable[Payload], n: int,
                  refetch: Optional[Callable[[int], Payload]] = None) -> ScanResult:
    """``payloads`` may be truncated (only the head of each message); a frame whose head ends before the
    picture data starts (for example a large SEI) is read again in full through ``refetch``."""
    keyframe = np.zeros(n, bool)
    carries = np.zeros(n, bool)
    decodable = np.full(n, NOT_DECODABLE, np.int8)
    stamps = np.full(n, -1, np.int64)
    b_frames = 0
    gaps_allowed: Optional[bool] = None
    first_params: Optional[int] = None
    ref_gaps: List[Tuple[int, int, int, int]] = []
    params = np.full(n, -1, np.int32)
    param_sets: List[bytes] = []
    param_rows: Dict[bytes, int] = {}

    def remember(i: int, prefix: bytes) -> None:          # distinct parameter sets, stored once
        if prefix:
            params[i] = param_rows.setdefault(prefix, len(param_sets))
            if params[i] == len(param_sets):
                param_sets.append(prefix)

    if codec == "h264":
        state, tracker, started = h264.H264State(), h264.FrameNumTracker(), False
        for i, p in enumerate(payloads):
            stamps[i] = -1 if p.stamp_ns is None else p.stamp_ns
            if not p.data:
                tracker.state = False if started else tracker.state
                continue
            u = h264.parse_unit(p.data, state)
            if not u.complete and p.truncated and refetch is not None:
                p = refetch(i)
                u = h264.parse_unit(p.data, state)
            keyframe[i] = u.keyframe
            carries[i] = u.has_sps and u.has_pps
            if carries[i] and first_params is None:
                first_params = i
            b_frames += u.slice_type == h264.SLICE_B
            gaps_allowed = u.gaps_allowed if u.gaps_allowed is not None else gaps_allowed
            if u.keyframe:
                if h264.params_available(u, state):
                    r = tracker.start(u)
                    started = True
                    remember(i, state.param_prefix(u.pps_id))
                else:
                    tracker.state, r = False, False
            elif not started:
                continue
            else:
                r = tracker.step(u)
                if tracker.gap is not None:
                    ref_gaps.append((i, *tracker.gap))
            decodable[i] = DECODABLE if r is True else (UNKNOWN if r is None else NOT_DECODABLE)
    elif codec == "h265":
        state5, started = h265.H265State(), False
        for i, p in enumerate(payloads):
            stamps[i] = -1 if p.stamp_ns is None else p.stamp_ns
            if not p.data:
                continue
            u = h265.parse_unit(p.data, state5)
            if not u.complete and p.truncated and refetch is not None:
                p = refetch(i)
                u = h265.parse_unit(p.data, state5)
            keyframe[i] = u.keyframe
            carries[i] = u.has_vps and u.has_sps and u.has_pps
            if carries[i] and first_params is None:
                first_params = i
            b_frames += u.slice_type == h265.SLICE_B
            if u.keyframe and h265.params_available(u, state5):
                started = True
                decodable[i] = DECODABLE
                remember(i, state5.param_prefix())
            elif started:
                decodable[i] = UNKNOWN                    # not inferred for H.265 yet
    else:                                                 # still images: every frame stands alone
        for i, p in enumerate(payloads):
            stamps[i] = -1 if p.stamp_ns is None else p.stamp_ns
            ok = bool(p.data)
            keyframe[i] = carries[i] = ok
            decodable[i] = DECODABLE if ok else NOT_DECODABLE
    return ScanResult(codec, keyframe, carries, decodable, b_frames, stamps, gaps_allowed, first_params,
                      ref_gaps, params, param_sets)
