"""H.265 headers: random access points (IRAP) and parameter sets.

Whether a frame after a missing picture can be decoded is not inferred for H.265 yet; callers get None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from mcap_frames.codecs.annexb import BitReader, iter_nal_units, rbsp

START_CODE = b"\x00\x00\x00\x01"
NAL_VPS, NAL_SPS, NAL_PPS = 32, 33, 34
SLICE_B, SLICE_P, SLICE_I = 0, 1, 2


@dataclass
class H265State:
    params: Dict[int, bytes] = field(default_factory=dict)       # NAL type → most recent NAL of that type
    extra_slice_header_bits: Dict[int, int] = field(default_factory=dict)   # pps id → num_extra_slice_header_bits

    def param_prefix(self, pps_id=None) -> bytes:
        return b"".join(START_CODE + self.params[t] for t in (NAL_VPS, NAL_SPS, NAL_PPS) if t in self.params)

    def copy(self) -> "H265State":
        return H265State(dict(self.params), dict(self.extra_slice_header_bits))


@dataclass
class H265Unit:
    keyframe: bool
    has_vps: bool
    has_sps: bool
    has_pps: bool
    complete: bool = True
    slice_type: Optional[int] = None   # of the picture's first slice segment; None when it could not be read


def _pps_extra_bits(nal: bytes):
    """(pps id, num_extra_slice_header_bits) from a PPS NAL unit."""
    b = BitReader(rbsp(nal[2:], 16))
    pps_id = b.ue()
    b.ue()                                          # sps id
    b.u(2)                                          # dependent_slice_segments_enabled, output_flag_present
    return pps_id, b.u(3)


def _slice_type(nal: bytes, nal_type: int, state: H265State) -> Optional[int]:
    """slice_type of a picture's first slice segment (B = 0, P = 1, I = 2), from its header."""
    b = BitReader(rbsp(nal[2:], 64))
    if not b.u(1):                                  # first_slice_segment_in_pic_flag
        return None
    if 16 <= nal_type <= 23:
        b.u(1)                                      # no_output_of_prior_pics_flag
    extra = state.extra_slice_header_bits.get(b.ue())
    if extra is None:                               # its PPS has not been seen
        return None
    b.u(extra)                                      # slice_reserved_flag[]
    return b.ue()


def parse_unit(data: bytes, state: H265State) -> H265Unit:
    irap = vps = sps = pps = picture = False
    slice_type = None
    for nal in iter_nal_units(data, lambda b: (b >> 1) & 0x3F < 32):
        if len(nal) < 2:
            continue
        t = (nal[0] >> 1) & 0x3F
        if t in (NAL_VPS, NAL_SPS, NAL_PPS):
            state.params[t] = nal
            vps, sps, pps = vps or t == NAL_VPS, sps or t == NAL_SPS, pps or t == NAL_PPS
            if t == NAL_PPS:
                try:
                    pps_id, extra = _pps_extra_bits(nal)
                    state.extra_slice_header_bits[pps_id] = extra
                except (IndexError, ValueError):
                    pass
        elif t < 32:                                    # picture data: parameter sets come before it
            irap, picture = 16 <= t <= 21, True
            try:
                slice_type = _slice_type(nal, t, state)
            except (IndexError, ValueError):
                pass
            break
    return H265Unit(irap, vps, sps, pps, picture, slice_type)


def params_available(unit: H265Unit, state: H265State) -> bool:
    return all(t in state.params for t in (NAL_VPS, NAL_SPS, NAL_PPS))
