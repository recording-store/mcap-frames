"""Opening a recording and finding its image / video streams."""

from __future__ import annotations

import os
from typing import BinaryIO, Dict, List, Union

from mcap_frames.container import Container
from mcap_frames.errors import TopicNotFound
from mcap_frames.messages import extractor_for
from mcap_frames.source import ReadStats, Source, open_source
from mcap_frames.video import VideoStream, scan_streams


class Recording:
    def __init__(self, src: Union[str, os.PathLike, BinaryIO, Source]):
        self.source = open_source(src)
        self._container = Container(self.source)
        self._streams: Dict[str, VideoStream] = {}

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def stats(self) -> ReadStats:
        """Bytes and reads so far (all streams of this recording)."""
        return self.source.stats

    @property
    def video_topics(self) -> List[str]:
        """Image / video topics with a supported message type that have at least one message."""
        counts = self._container.statistics.channel_message_counts if self._container.statistics else None
        out = []
        for ch in self._container.channels.values():
            if extractor_for(self._container.schemas.get(ch.schema_id), ch) is None:
                continue
            has = counts.get(ch.id, 0) > 0 if counts is not None else bool(self._container.chunks_with(ch.id))
            if has:
                out.append(ch.topic)
        return sorted(out)

    def video(self, topic: str) -> VideoStream:
        if topic not in self._streams:
            channels = [c for c in self._container.channels.values() if c.topic == topic]
            if not channels:
                raise TopicNotFound(f"no topic {topic!r} in {self.name}")
            ch = channels[0]
            schema = self._container.schemas.get(ch.schema_id)
            extract = extractor_for(schema, ch)
            if extract is None:
                raise TopicNotFound(f"{topic!r} is {schema.name if schema else 'schemaless'!r} "
                                    f"({ch.message_encoding}), not a supported image / video message")
            self._streams[topic] = VideoStream(self._container, ch, schema, extract)
        return self._streams[topic]

    def scan(self, topics=None) -> None:
        """Read the headers of every frame of several topics (all by default) in one pass over the file, each
        chunk read once. Optional: decodable(), keyframes, stamps_ns and health checks scan on first use, but
        one topic at a time."""
        scan_streams([self.video(t) for t in (topics or self.video_topics)])

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open(src: Union[str, os.PathLike, BinaryIO, Source]) -> Recording:  # noqa: A001 (mirrors builtins.open)
    return Recording(src)
