"""Frame-accurate video access for MCAP files.

    import mcap_frames as mf
    rec = mf.open("episode.mcap")              # or an http(s) URL with range support
    v = rec.video("/camera/front")
    f = v.frame(120)                           # frame 120 = the 120th message of the topic
    report = mf.check("episode.mcap")          # damaged frames, skipped frames, timestamps
"""

from mcap_frames.errors import (
    FrameNotDecodable, McapFramesError, NoKeyframeBefore, TopicNotFound, UnsupportedFile, UnsupportedStream,
)
from mcap_frames.export import export_frames, to_mp4
from mcap_frames.health import Issue, Report, TopicReport, check
from mcap_frames import recording as _recording
from mcap_frames.recording import Recording
from mcap_frames.source import FileSource, HttpSource, Source
from mcap_frames.video import EncodedRun, Frame, VideoStream

__version__ = "0.1.0"

open = _recording.open        # mf.open(...); left out of __all__ so that ``import *`` keeps the builtin open
__all__ = [
    "check", "export_frames", "to_mp4",
    "Recording", "VideoStream", "Frame", "EncodedRun", "Report", "TopicReport", "Issue",
    "Source", "FileSource", "HttpSource",
    "McapFramesError", "UnsupportedFile", "TopicNotFound", "UnsupportedStream", "NoKeyframeBefore",
    "FrameNotDecodable",
]
