"""Exceptions raised by mcap-frames."""


class McapFramesError(Exception):
    """Base class for all errors raised by this package."""


class UnsupportedFile(McapFramesError):
    """The file cannot be read by index (for example it has no summary section or no chunk index)."""


class TopicNotFound(McapFramesError, KeyError):
    """The topic does not exist, or it is not an image / video topic this package understands."""

    def __str__(self) -> str:                      # KeyError would otherwise quote the message
        return str(self.args[0]) if self.args else ""


class UnsupportedStream(McapFramesError):
    """The stream cannot be accessed frame by frame (unknown codec, or B-frames reorder pictures)."""


class NoKeyframeBefore(McapFramesError):
    """There is no keyframe at or before the requested frame (the recording starts mid-GOP)."""


class FrameNotDecodable(McapFramesError):
    """A frame the requested frame depends on is missing, so it cannot be decoded correctly."""
