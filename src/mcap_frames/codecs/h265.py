"""H.265 headers: random access points (IRAP) and parameter sets.

Whether a frame after a missing picture can be decoded is not inferred for H.265 yet; callers get None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from mcap_frames.codecs.annexb import iter_nal_units

START_CODE = b"\x00\x00\x00\x01"
NAL_VPS, NAL_SPS, NAL_PPS = 32, 33, 34


@dataclass
class H265State:
    params: Dict[int, bytes] = field(default_factory=dict)       # NAL type → most recent NAL of that type

    def param_prefix(self, pps_id=None) -> bytes:
        return b"".join(START_CODE + self.params[t] for t in (NAL_VPS, NAL_SPS, NAL_PPS) if t in self.params)

    def copy(self) -> "H265State":
        return H265State(dict(self.params))


@dataclass
class H265Unit:
    keyframe: bool
    has_vps: bool
    has_sps: bool
    has_pps: bool
    complete: bool = True


def parse_unit(data: bytes, state: H265State) -> H265Unit:
    irap = vps = sps = pps = picture = False
    for nal in iter_nal_units(data, lambda b: (b >> 1) & 0x3F < 32):
        if len(nal) < 2:
            continue
        t = (nal[0] >> 1) & 0x3F
        if t in (NAL_VPS, NAL_SPS, NAL_PPS):
            state.params[t] = nal
            vps, sps, pps = vps or t == NAL_VPS, sps or t == NAL_SPS, pps or t == NAL_PPS
        elif t < 32:                                    # picture data: parameter sets come before it
            irap, picture = 16 <= t <= 21, True
            break
    return H265Unit(irap, vps, sps, pps, picture)


def params_available(unit: H265Unit, state: H265State) -> bool:
    return all(t in state.params for t in (NAL_VPS, NAL_SPS, NAL_PPS))
