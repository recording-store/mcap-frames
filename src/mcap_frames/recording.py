"""Opening a recording and finding its image / video streams."""

from __future__ import annotations

import os
from typing import BinaryIO, Dict, List, Optional, Union

from mcap_frames.container import Container
from mcap_frames.errors import TopicNotFound
from mcap_frames.messages import extractor_for
from mcap_frames.source import ReadStats, Source, open_source
from mcap_frames.video import VideoStream, scan_streams


class Recording:
    """An MCAP recording and its image / video streams.

    ``index``: an index file (path or http(s) URL) written by save_index. When it matches the recording, opening
    reads nothing but the end of the recording; otherwise the recording is read as if there were no index.
    ``index_status`` says which: "none" (no index given), "loaded", "missing" or "stale: <reason>"."""

    def __init__(self, src: Union[str, os.PathLike, BinaryIO, Source], index=None, sha256: Optional[str] = None):
        from mcap_frames import indexfile

        self.source = open_source(src)
        self._streams: Dict[str, VideoStream] = {}
        self._stored = None
        self._sha256 = sha256
        self.index_status = "none"
        if index is not None:
            data = indexfile.read_bytes(index)
            if data is None:
                self.index_status = "missing"
            else:
                try:
                    self._stored = indexfile.load(data, self.source, sha256)
                    self.index_status = "loaded"
                except indexfile.StaleIndex as e:
                    self.index_status = f"stale: {e}"
        self._container = self._stored.container if self._stored else Container(self.source)
        self._full: Optional[Container] = None if self._stored else self._container

    def _full_container(self) -> Container:
        """The container read from the recording itself (for topics an index file does not cover)."""
        if self._full is None:
            self._full = Container(self.source)
        return self._full

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
        if self._stored:
            return sorted(t for t, st in self._stored.topics.items() if len(st.index))
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
            stored = self._stored.topics.get(topic) if self._stored else None
            container = self._container if stored else self._full_container()
            channels = ([container.channels[stored.channel_id]] if stored
                        else [c for c in container.channels.values() if c.topic == topic])
            if not channels:
                raise TopicNotFound(f"no topic {topic!r} in {self.name}")
            ch = channels[0]
            schema = container.schemas.get(ch.schema_id)
            extract = extractor_for(schema, ch)
            if extract is None:
                raise TopicNotFound(f"{topic!r} is {schema.name if schema else 'schemaless'!r} "
                                    f"({ch.message_encoding}), not a supported image / video message")
            v = VideoStream(container, ch, schema, extract)
            if stored:
                v._index, v._codec, v._scan = stored.index, stored.codec, stored.scan
            self._streams[topic] = v
        return self._streams[topic]

    def save_index(self, path, scan: bool = True, sha256: Optional[str] = None) -> None:
        """Write an index file for every video topic: where each frame is, and (``scan``, the default) what the
        scan found. Nothing is judged: health checks are recomputed from these facts. Written atomically.

        ``sha256``: the recording's hash, when known; stored, and compared when the index is loaded with one."""
        from mcap_frames import indexfile
        from mcap_frames.errors import McapFramesError

        topics = self.video_topics
        if scan:
            try:
                self.scan(topics)                          # one pass for all topics
            except McapFramesError:
                for t in topics:                           # one bad topic: scan the others one by one
                    try:
                        self.video(t)._scanned()
                    except McapFramesError:
                        pass                               # stored unscanned
        streams = [self.video(t) for t in topics]
        indexfile.write(path, self._container, streams, sha256 or self._sha256)

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


def open(src: Union[str, os.PathLike, BinaryIO, Source], index=None,  # noqa: A001 (mirrors builtins.open)
         sha256: Optional[str] = None) -> Recording:
    return Recording(src, index, sha256)
