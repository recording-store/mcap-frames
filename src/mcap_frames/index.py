"""Frame index of one channel, built from the file's own MessageIndex records (no message bytes read).

Frame n is the n-th message of the channel in the order it was written, which is the order a decoder needs.
log_time usually increases with it, but not always (clock jitter); lookups by time use a sorted copy. The
numbering never depends on what a decoder manages to output, so it stays stable when the stream is damaged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from mcap_frames.container import Container, MessageRef


@dataclass
class FrameIndex:
    refs: List[MessageRef]
    log_time_ns: np.ndarray            # per frame, in frame order (may decrease where the clock jittered)

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def time_order(self) -> np.ndarray:
        """Frame numbers sorted by log_time (stable), for lookups by time."""
        if not hasattr(self, "_time_order"):
            self._time_order = np.argsort(self.log_time_ns, kind="stable")
        return self._time_order


def build_index(container: Container, channel_id: int) -> FrameIndex:
    refs = container.refs(channel_id)
    return FrameIndex(refs, np.fromiter((r.log_time for r in refs), dtype=np.int64, count=len(refs)))
