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


class _NotRetried(IOError):
    """A failure that retrying cannot fix."""


class HttpSource(Source):
    """An http(s) URL that supports range requests (object storage, presigned URLs, static file servers).

    - Each reading thread keeps one persistent connection per host and reuses it.
    - The size comes from a one-byte range GET, not HEAD: presigned URLs are signed for one method.
    - Connection errors, timeouts, 408 / 429 and 5xx answers are retried with exponential backoff; other
      answers (404, 403 ...) fail at once. Redirects are followed.
    - A server that answers a range request with the whole file is refused (the body is not downloaded).
    - Authorization and Cookie headers are not sent on to another host when a redirect leaves the original one.
    Proxies from the environment are not used."""

    # Measured with 10 ms latency and 50 MB/s per connection: with 16 requests in flight a round trip costs
    # less than reading 64 KB more, and larger gaps end up reading whole files.
    merge_gap = 64 * 1024
    concurrency = 16
    _RETRY_STATUS = {408, 429}

    def __init__(self, url: str, headers: dict | None = None, connect_timeout: float = 10.0,
                 read_timeout: float = 30.0, retries: int = 3, backoff: float = 0.2, max_redirects: int = 5):
        super().__init__()
        self.name = url
        self._url = url
        self._headers = dict(headers or {})
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        self.retries, self.backoff, self.max_redirects = retries, backoff, max_redirects
        self._local = threading.local()
        self._conns: List[object] = []                     # every connection, so close() can reach them all
        self._conns_lock = threading.Lock()
        _body, headers_ = self._get(0, 0)
        content_range = headers_.get("Content-Range", "")
        self.etag = headers_.get("ETag")
        try:
            self.size = int(content_range.rsplit("/", 1)[1])
        except (IndexError, ValueError):
            self.size = self._size_from_head()             # servers that omit Content-Range on 206

    def _size_from_head(self) -> int:
        from urllib.parse import urlsplit
        parts = urlsplit(self._url)
        conn = self._connection(parts.scheme, parts.netloc)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            conn.request("HEAD", target, headers=self._headers)
            resp = conn.getresponse()
            resp.read()
            if resp.will_close:
                self._drop(parts.scheme, parts.netloc)
        except BaseException:
            self._drop(parts.scheme, parts.netloc)
            raise
        length = resp.getheader("Content-Length")
        if resp.status != 200 or length is None:
            raise IOError(f"{self.name} reported neither Content-Range nor Content-Length (HEAD status {resp.status})")
        return int(length)

    # -------------------------------------------------------------- connections
    def _connection(self, scheme: str, host: str):
        import http.client
        conns = self._local.__dict__.setdefault("conns", {})
        conn = conns.get((scheme, host))
        if conn is None:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            conn = cls(host, timeout=self.connect_timeout)
            conn.connect()
            conn.sock.settimeout(self.read_timeout)
            conns[(scheme, host)] = conn
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def _drop(self, scheme: str, host: str) -> None:
        conn = self._local.__dict__.get("conns", {}).pop((scheme, host), None)
        if conn is not None:
            conn.close()
            with self._conns_lock:
                if conn in self._conns:
                    self._conns.remove(conn)

    def _request(self, start: int, end: int):
        """One GET with a Range header, following redirects. Returns (status, headers, body)."""
        from urllib.parse import urljoin, urlsplit
        url = self._url
        origin = urlsplit(self.name).netloc
        for _ in range(self.max_redirects + 1):
            parts = urlsplit(url)
            conn = self._connection(parts.scheme, parts.netloc)
            target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            headers = self._headers if parts.netloc == origin else {
                k: v for k, v in self._headers.items() if k.lower() not in ("authorization", "cookie")}
            try:
                conn.request("GET", target, headers=dict(headers, Range=f"bytes={start}-{end}"))
                resp = conn.getresponse()
                if resp.status in (301, 302, 303, 307, 308):
                    resp.read()
                    url = urljoin(url, resp.getheader("Location", ""))
                    continue
                if resp.status == 200:                     # whole file instead of a range: do not download it
                    self._drop(parts.scheme, parts.netloc)
                    return resp.status, resp.headers, b""
                body = resp.read()
                if resp.will_close:
                    self._drop(parts.scheme, parts.netloc)
                if resp.status == 206:
                    self._url = url                         # skip the redirect next time
                return resp.status, resp.headers, body
            except BaseException:
                self._drop(parts.scheme, parts.netloc)
                raise
        raise _NotRetried(f"{self.name}: more than {self.max_redirects} redirects")

    def _get(self, start: int, end: int):
        import http.client
        import time
        last: Exception = IOError("no attempt made")
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                status, headers, body = self._request(start, end)
            except _NotRetried:
                raise
            except (OSError, http.client.HTTPException) as e:   # connection reset, timeout, broken response
                last = e if str(e) else IOError(type(e).__name__)
                continue
            if status == 206:
                return body, headers
            if status == 200:
                raise IOError(f"{self.name} ignored the range request (status 200): the server does not "
                              "support range requests")
            if status in self._RETRY_STATUS or status >= 500:
                last = IOError(f"HTTP {status}")
                continue
            raise IOError(f"{self.name}: HTTP {status}")
        raise IOError(f"{self.name}: failed after {self.retries + 1} attempts: {last}")

    def _read(self, offset: int, size: int) -> bytes:
        body, _headers = self._get(offset, offset + size - 1)
        return body

    def close(self) -> None:
        super().close()
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            conn.close()


def open_source(src: Union[str, os.PathLike, BinaryIO, Source]) -> Source:
    if isinstance(src, Source):
        return src
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        return HttpSource(src)
    return FileSource(src)


def total_size(ranges: Iterable[Tuple[int, int]]) -> int:
    return sum(size for _off, size in ranges)
