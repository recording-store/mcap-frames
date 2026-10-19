"""H.264 headers: keyframes, parameter sets, slice type, and frame_num.

frame_num is what makes "which frames can be decoded" knowable without decoding: every reference picture
increments it by one (modulo MaxFrameNum), so a jump means a reference picture is missing and every frame
up to the next keyframe is decoded against the wrong reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mcap_frames.codecs.annexb import BitReader, iter_nal_units, rbsp

START_CODE = b"\x00\x00\x00\x01"
NAL_SLICE, NAL_IDR, NAL_SEI, NAL_SPS, NAL_PPS = 1, 5, 6, 7, 8
SLICE_P, SLICE_B, SLICE_I, SLICE_SP, SLICE_SI = 0, 1, 2, 3, 4
_HIGH_PROFILES = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}


@dataclass
class Sps:
    sps_id: int
    log2_max_frame_num: int
    separate_colour_plane: bool
    gaps_allowed: bool


@dataclass
class H264State:
    """What earlier access units told us: parameter sets by id, and their raw NAL units for re-sending."""

    sps: Dict[int, Sps] = field(default_factory=dict)
    pps_to_sps: Dict[int, int] = field(default_factory=dict)
    sps_nal: Dict[int, bytes] = field(default_factory=dict)
    pps_nal: Dict[int, bytes] = field(default_factory=dict)

    def param_prefix(self, pps_id: Optional[int] = None) -> bytes:
        """The SPS and PPS to put in front of a keyframe that does not carry them."""
        if pps_id is not None and pps_id in self.pps_nal:
            sps_id = self.pps_to_sps.get(pps_id)
            nals = [self.sps_nal.get(sps_id), self.pps_nal[pps_id]]
        else:
            nals = list(self.sps_nal.values()) + list(self.pps_nal.values())
        return b"".join(START_CODE + n for n in nals if n)

    def copy(self) -> "H264State":
        return H264State(dict(self.sps), dict(self.pps_to_sps), dict(self.sps_nal), dict(self.pps_nal))


@dataclass
class H264Unit:
    keyframe: bool                 # decoding can start here: IDR, or an I slice with a recovery point of 0
    idr: bool
    has_sps: bool
    has_pps: bool
    slice_type: Optional[int]      # of the first slice; None when no slice could be parsed
    frame_num: Optional[int]
    ref_idc: Optional[int]
    pps_id: Optional[int]
    gaps_allowed: Optional[bool]
    max_frame_num: Optional[int]

    @property
    def complete(self) -> bool:
        """A slice was found (False when the data given was cut before the picture data started)."""
        return self.slice_type is not None


def parse_sps(nal: bytes) -> Sps:
    b = BitReader(rbsp(nal[1:]))
    profile = b.u(8)
    b.u(16)                                         # constraint flags, level
    sps_id = b.ue()
    separate = False
    if profile in _HIGH_PROFILES:
        chroma = b.ue()
        if chroma == 3:
            separate = bool(b.u(1))
        b.ue()                                      # bit_depth_luma
        b.ue()                                      # bit_depth_chroma
        b.u(1)                                      # qpprime_y_zero_transform_bypass
        if b.u(1):                                  # seq_scaling_matrix_present
            for i in range(8 if chroma != 3 else 12):
                if b.u(1):
                    size, last, nxt = (16 if i < 6 else 64), 8, 8
                    for _ in range(size):
                        if nxt:
                            nxt = (last + b.se() + 256) % 256
                        last = nxt or last
    log2_max_frame_num = b.ue() + 4
    poc_type = b.ue()
    if poc_type == 0:
        b.ue()
    elif poc_type == 1:
        b.u(1)
        b.se()
        b.se()
        for _ in range(b.ue()):
            b.se()
    b.ue()                                          # max_num_ref_frames
    gaps = bool(b.u(1))
    return Sps(sps_id, log2_max_frame_num, separate, gaps)


def _pps_ids(nal: bytes):
    b = BitReader(rbsp(nal[1:], 16))
    return b.ue(), b.ue()                           # pps_id, sps_id


def _recovery_point_now(nal: bytes) -> bool:
    """SEI with a recovery point whose recovery_frame_cnt is 0 (the picture itself is clean)."""
    data = rbsp(nal[1:], 4096)
    i = 0
    try:
        while i < len(data) and data[i] != 0x80:
            ptype = 0
            while data[i] == 0xFF:
                ptype += 255
                i += 1
            ptype += data[i]
            i += 1
            size = 0
            while data[i] == 0xFF:
                size += 255
                i += 1
            size += data[i]
            i += 1
            if ptype == 6:
                return BitReader(data[i:i + size]).ue() == 0
            i += size
    except (IndexError, ValueError):
        return False
    return False


def _is_slice(first: int) -> bool:
    return first & 0x1F in (NAL_SLICE, NAL_IDR)


def parse_unit(data: bytes, state: H264State) -> H264Unit:
    """Parse one access unit (one message), updating ``state`` with any parameter sets it carries.

    Parameter sets and SEI come before the picture data, so parsing stops at the first slice: the rest of
    the access unit is never scanned. ``data`` may therefore be just the first few kilobytes."""
    idr = has_sps = has_pps = recovery = False
    slice_type = frame_num = ref_idc = pps_id = None
    sps: Optional[Sps] = None
    for nal in iter_nal_units(data, _is_slice):
        t = nal[0] & 0x1F
        if t == NAL_SPS:
            has_sps = True
            try:
                s = parse_sps(nal)
                state.sps[s.sps_id] = s
                state.sps_nal[s.sps_id] = nal
            except (IndexError, ValueError):
                pass
        elif t == NAL_PPS:
            has_pps = True
            try:
                p, s = _pps_ids(nal)
                state.pps_to_sps[p] = s
                state.pps_nal[p] = nal
            except (IndexError, ValueError):
                pass
        elif t == NAL_SEI:
            recovery = recovery or _recovery_point_now(nal)
        elif t in (NAL_SLICE, NAL_IDR):
            idr = t == NAL_IDR
            ref_idc = (nal[0] >> 5) & 3
            try:
                b = BitReader(rbsp(nal[1:], 64))
                b.ue()                                  # first_mb_in_slice
                slice_type = b.ue() % 5
                pps_id = b.ue()
                sps = state.sps.get(state.pps_to_sps.get(pps_id, -1))
                if sps is not None:
                    if sps.separate_colour_plane:
                        b.u(2)
                    frame_num = b.u(sps.log2_max_frame_num)
            except (IndexError, ValueError):
                pass
            break                                       # header fields come from the first slice
    keyframe = idr or (recovery and slice_type in (SLICE_I, SLICE_SI))
    return H264Unit(keyframe, idr, has_sps, has_pps, slice_type, frame_num, ref_idc, pps_id,
                    sps.gaps_allowed if sps else None, (1 << sps.log2_max_frame_num) if sps else None)


def params_available(unit: H264Unit, state: H264State) -> bool:
    """Whether the parameter sets this unit's slices need are known (carried here or seen earlier)."""
    if unit.pps_id is None:
        return unit.has_sps and unit.has_pps
    sps_id = state.pps_to_sps.get(unit.pps_id)
    return sps_id is not None and sps_id in state.sps_nal


class FrameNumTracker:
    """Follows frame_num through a run of access units starting at a keyframe.

    ``step`` returns True while every picture's references are present, False after a gap, and None when the
    stream allows gaps (gaps_in_frame_num_allowed) so a gap does not prove anything. ``gap`` is set on the step
    that found one: (expected frame_num, frame_num found, max_frame_num).
    """

    def __init__(self) -> None:
        self.expect: Optional[int] = None
        self.state: Optional[bool] = None
        self.gap: Optional[Tuple[int, int, int]] = None

    def start(self, unit: H264Unit) -> Optional[bool]:
        self.state = True if unit.frame_num is not None else None
        self._advance(unit)
        return self.state

    def step(self, unit: H264Unit) -> Optional[bool]:
        self.gap = None
        if unit.keyframe:
            return self.start(unit)
        if self.state is True:
            if unit.frame_num is None or self.expect is None:
                self.state = None
            elif unit.frame_num != self.expect:
                self.state = None if unit.gaps_allowed else False
                if self.state is False and unit.max_frame_num:
                    self.gap = (self.expect, unit.frame_num, unit.max_frame_num)
        self._advance(unit)
        return self.state

    def _advance(self, unit: H264Unit) -> None:
        if unit.frame_num is None or unit.max_frame_num is None:
            return
        self.expect = (unit.frame_num + 1) % unit.max_frame_num if unit.ref_idc else unit.frame_num


def with_prefix(data: bytes, prefix: bytes) -> bytes:
    return prefix + data if prefix else data


__all__: List[str] = ["H264State", "H264Unit", "parse_unit", "params_available", "FrameNumTracker", "with_prefix"]
