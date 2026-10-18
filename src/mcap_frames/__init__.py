"""Frame-accurate video access for MCAP files.

    import mcap_frames as mf
    rec = mf.open("episode.mcap")              # or an http(s) URL with range support
    v = rec.video("/camera/front")
    f = v.frame(120)                           # frame 120 = the 120th message of the topic
    report = mf.check("episode.mcap")          # keyframes, missing references, timestamp gaps
"""

from mcap_frames.errors import (
    FrameNotDecodable, McapFramesError, NoKeyframeBefore, TopicNotFound, UnsupportedFile, UnsupportedStream,
)
from mcap_frames.export import export_frames, to_mp4
from mcap_frames.health import Issue, Report, TopicReport, check
from mcap_frames.recording import Recording, open
from mcap_frames.source import FileSource, HttpSource, Source
from mcap_frames.video import EncodedRun, Frame, VideoStream

__version__ = "0.1.0.dev0"

__all__ = [
    "open", "check", "export_frames", "to_mp4",
    "Recording", "VideoStream", "Frame", "EncodedRun", "Report", "TopicReport", "Issue",
    "Source", "FileSource", "HttpSource",
    "McapFramesError", "UnsupportedFile", "TopicNotFound", "UnsupportedStream", "NoKeyframeBefore",
    "FrameNotDecodable",
]
