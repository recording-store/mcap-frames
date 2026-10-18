"""Byte sources: where the MCAP bytes come from.

Every read goes through ``read(offset, size)`` and is counted, so callers can see how many bytes and
requests an operation cost. ``read_many`` merges nearby ranges into one read and issues the resulting reads
concurrently. With requests in flight at once, the round trip of an extra request is mostly hidden, so only
ranges less than 64 KB apart are merged; merging more ends up reading picture data nobody asked for.
"""

from __future__ import annotations

import io
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import BinaryIO, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass
class ReadStats:
    bytes: int = 0
    reads: int = 0


class Source:
    """Random-access, read-only bytes."""

    name: str = ""
    size: int = 0
    merge_gap: int = 64 * 1024        # ranges closer than this are fetched with one read
    concurrency: int = 1              # reads read_many may have in flight at once

    def __init__(self) -> None:
        self.stats = ReadStats()
        self._stats_lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None

    def _read(self, offset: int, size: int) -> bytes:
        raise NotImplementedError

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError(f"read {offset}+{size} is outside {self.name} ({self.size} bytes)")
        if size == 0:
            return b""
        data = self._read(offset, size)
        if len(data) != size:
            raise IOError(f"short read from {self.name}: wanted {size} bytes at {offset}, got {len(data)}")
        with self._stats_lock:
            self.stats.bytes += size
            self.stats.reads += 1
        return data

    def read_many(self, ranges: Sequence[Tuple[int, int]], merge_gap: Optional[int] = None) -> List[bytes]:
        """Read several (offset, size) ranges; returns them in the order given."""
        gap = self.merge_gap if merge_gap is None else merge_gap
        order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
        groups: List[List[int]] = []
        group_end = 0
        for i in order:
            off, size = ranges[i]
            if not groups or off > group_end + gap:
                groups.append([])
                group_end = 0
            groups[-1].append(i)
            group_end = max(group_end, off + size)

        def fetch(group: List[int]) -> bytes:
            start = ranges[group[0]][0]
            end = max(ranges[i][0] + ranges[i][1] for i in group)
            return self.read(start, end - start)

        if self.concurrency > 1 and len(groups) > 1:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.concurrency, thread_name_prefix="mcap-frames-read")
            blobs = list(self._pool.map(fetch, groups))
        else:
            blobs = [fetch(g) for g in groups]
        out: List[bytes] = [b""] * len(ranges)
        for group, blob in zip(groups, blobs):
            start = ranges[group[0]][0]
            for i in group:
                off, size = ranges[i]
                out[i] = blob[off - start:off - start + size]
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


class FileSource(Source):
    """A local file, or any seekable binary file object."""

    merge_gap = 64 * 1024
    concurrency = 8

    def __init__(self, file: Union[str, os.PathLike, BinaryIO]):
        super().__init__()
        self._lock = threading.Lock()
        if isinstance(file, (str, os.PathLike)):
            self._fh = open(file, "rb")
            self._owned = True
            self.name = os.fspath(file)
        else:
            self._fh = file
            self._owned = False
            self.name = getattr(file, "name", "<file object>")
        self._fh.seek(0, os.SEEK_END)
        self.size = self._fh.tell()
        try:
            self._fd: Optional[int] = self._fh.fileno() if hasattr(os, "pread") else None
        except (AttributeError, OSError, io.UnsupportedOperation):
            self._fd = None                             # in-memory file objects: no descriptor

    def _read(self, offset: int, size: int) -> bytes:
        if self._fd is not None:                        # positional read: safe from several threads at once
            parts, done = [], 0
            while done < size:
                chunk = os.pread(self._fd, size - done, offset + done)
                if not chunk:
                    break
                parts.append(chunk)
                done += len(chunk)
            return b"".join(parts)
        with self._lock:
            self._fh.seek(offset)
            return self._fh.read(size)

    def close(self) -> None:
        super().close()
        if self._owned:
            self._fh.close()


class HttpSource(Source):
    """An http(s) URL that supports range requests (object storage, presigned URLs, static file servers)."""

    # Measured with 10 ms latency and 50 MB/s per connection: with 16 requests in flight a round trip costs
    # less than reading 64 KB more, and larger gaps end up reading whole files (see benchmarks).
    merge_gap = 64 * 1024
    concurrency = 16

    def __init__(self, url: str, headers: dict | None = None, timeout: float = 30.0):
        super().__init__()
        self.name = url
        self._headers = dict(headers or {})
        self._timeout = timeout
        req = urllib.request.Request(url, method="HEAD", headers=self._headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            if length is None:
                raise IOError(f"{url} did not report Content-Length")
            self.size = int(length)
            self.etag = resp.headers.get("ETag")

    def _read(self, offset: int, size: int) -> bytes:
        headers = dict(self._headers, Range=f"bytes={offset}-{offset + size - 1}")
        req = urllib.request.Request(self.name, headers=headers)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            if resp.status != 206:
                raise IOError(f"{self.name} ignored the range request (status {resp.status})")
            return resp.read()


def open_source(src: Union[str, os.PathLike, BinaryIO, Source]) -> Source:
    if isinstance(src, Source):
        return src
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        return HttpSource(src)
    return FileSource(src)


def total_size(ranges: Iterable[Tuple[int, int]]) -> int:
    return sum(size for _off, size in ranges)
