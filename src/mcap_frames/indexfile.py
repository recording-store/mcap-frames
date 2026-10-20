"""Index files: what opening and scanning a recording found, stored so the next open need not read it again.

Format: see docs/index-file.md. One NumPy ``.npz`` archive per recording, read with ``allow_pickle=False``. An index
file that does not match its recording, or fails any check here, is treated as stale and not used.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from mcap.records import Channel, Schema

from mcap_frames import scan as scanmod
from mcap_frames.container import Container, MessageRef, StoredChunk, _LooseMessages
from mcap_frames.errors import McapFramesError
from mcap_frames.index import FrameIndex
from mcap_frames.source import Source

FORMAT = "mcap-frames-index"
VERSION = 1
TAIL = 64 * 1024                  # bytes at the end of the recording that identify it

KEYFRAME, CARRIES_PARAMS = 1, 2   # bits of the per-frame flags


class StaleIndex(McapFramesError):
    """The index file cannot be used for this recording (the reason is the message)."""


def fingerprint(source: Source, sha256: Optional[str] = None) -> dict:
    """What identifies the recording: its size, a hash of its end, and the server's ETag when there is one."""
    n = min(source.size, TAIL)
    return {"size": source.size, "tail_sha256": hashlib.sha256(source.read(source.size - n, n)).hexdigest(),
            "etag": getattr(source, "etag", None), "sha256": sha256}


def _mismatch(stored: dict, current: dict) -> Optional[str]:
    if stored.get("size") != current["size"]:
        return f"the recording is {current['size']} bytes, the index was made for {stored.get('size')}"
    if stored.get("tail_sha256") != current["tail_sha256"]:
        return "the end of the recording differs from when the index was made"
    for key in ("etag", "sha256"):
        if stored.get(key) and current.get(key) and stored[key] != current[key]:
            return f"the recording's {key} differs from when the index was made"
    return None


@dataclass
class StoredTopic:
    topic: str
    channel_id: int
    codec: Optional[str]
    index: FrameIndex
    scan: Optional[scanmod.ScanResult]


@dataclass
class StoredIndex:
    container: Container
    topics: Dict[str, StoredTopic] = field(default_factory=dict)


# ---------------------------------------------------------------- writing
def write(path, container: Container, streams: List, sha256: Optional[str] = None) -> None:
    """Write the index of ``streams`` (VideoStreams of ``container``) to ``path``, atomically."""
    from mcap_frames import __version__

    arrays: Dict[str, np.ndarray] = {}
    chunks = container.chunks
    arrays["chunks/start"] = np.array([0 if isinstance(c, _LooseMessages) else c.chunk_start_offset for c in chunks],
                                      np.uint64)
    arrays["chunks/records_start"] = np.array([container._records_start(k) for k in range(len(chunks))], np.uint64)
    arrays["chunks/compressed_size"] = np.array([c.compressed_size for c in chunks], np.uint64)
    arrays["chunks/uncompressed_size"] = np.array([c.uncompressed_size for c in chunks], np.uint64)
    arrays["chunks/compression"] = np.array([c.compression or "" for c in chunks], dtype="<U8")
    arrays["chunks/kind"] = np.array([1 if isinstance(c, _LooseMessages) else 0 for c in chunks], np.uint8)

    topics = []
    for i, v in enumerate(streams):
        key = f"t{i}"
        refs = v._frame_index.refs
        arrays[f"{key}/chunk"] = np.array([r.chunk for r in refs], np.uint32)
        arrays[f"{key}/offset"] = np.array([r.offset for r in refs], np.uint64)
        arrays[f"{key}/bound"] = np.array([r.bound for r in refs], np.uint64)
        arrays[f"{key}/log_time"] = np.array([r.log_time for r in refs], np.uint64)
        try:
            codec = v.codec if len(v) else None
        except McapFramesError:
            codec = None
        s = v._scan
        entry = {"key": key, "topic": v.topic, "channel_id": v.channel.id, "frames": len(refs), "codec": codec,
                 "scanned": s is not None}
        if s is not None:
            arrays[f"{key}/stamp"] = s.stamp_ns.astype(np.int64)
            arrays[f"{key}/flags"] = (s.keyframe * KEYFRAME + s.carries_params * CARRIES_PARAMS).astype(np.uint8)
            arrays[f"{key}/decodable"] = s.decodable.astype(np.int8)
            arrays[f"{key}/params"] = (s.params if s.params is not None else np.full(len(refs), -1)).astype(np.int32)
            sizes = np.array([len(p) for p in s.param_sets], np.uint64)
            arrays[f"{key}/param_data"] = np.frombuffer(b"".join(s.param_sets), np.uint8)
            arrays[f"{key}/param_offset"] = (np.cumsum(sizes) - sizes).astype(np.uint64)
            arrays[f"{key}/param_size"] = sizes
            entry.update(b_frames=int(s.b_frames), gaps_allowed=s.gaps_allowed, params_first_seen=s.params_first_seen,
                         reference_gaps=[list(g) for g in s.reference_gaps])
        topics.append(entry)

    meta = {
        "format": FORMAT, "version": VERSION, "created_by": f"mcap-frames {__version__}",
        "file": fingerprint(container.source, sha256),
        "container": {"recovered": container.recovered, "unchunked_messages": container.unchunked_messages,
                      "ignored_tail_bytes": container.ignored_tail_bytes,
                      "has_message_indexes": container.has_message_indexes},
        "schemas": [{"id": sc.id, "name": sc.name, "encoding": sc.encoding,
                     "data_b64": base64.b64encode(sc.data).decode("ascii")} for sc in container.schemas.values()],
        "channels": [{"id": ch.id, "topic": ch.topic, "schema_id": ch.schema_id,
                      "message_encoding": ch.message_encoding, "metadata": dict(ch.metadata)}
                     for ch in container.channels.values()],
        "topics": topics,
    }
    arrays["meta"] = np.array(json.dumps(meta))

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------- reading
def read_bytes(location) -> Optional[bytes]:
    """The index file's bytes, or None when there is none. ``location``: a path or an http(s) URL."""
    if isinstance(location, str) and location.startswith(("http://", "https://")):
        from mcap_frames.source import HttpSource
        try:
            src = HttpSource(location)
        except IOError as e:
            if "HTTP 404" in str(e):
                return None
            raise
        try:
            return src.read(0, src.size)
        finally:
            src.close()
    try:
        with open(location, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def load(data: bytes, source: Source, sha256: Optional[str] = None) -> StoredIndex:
    """Rebuild the container and topics from an index file's bytes; raises StaleIndex when it cannot be used."""
    try:
        npz = np.load(io.BytesIO(data), allow_pickle=False)
        a = {k: npz[k] for k in npz.files}
        meta = json.loads(str(a["meta"]))
    except Exception as e:                                 # not an npz, or damaged
        raise StaleIndex(f"unreadable index file ({type(e).__name__})") from None
    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise StaleIndex(f"index file is {meta.get('format')} version {meta.get('version')}, "
                         f"this library reads {FORMAT} version {VERSION}")
    reason = _mismatch(meta.get("file", {}), fingerprint(source, sha256))
    if reason:
        raise StaleIndex(reason)
    try:
        return _rebuild(meta, a, source)
    except StaleIndex:
        raise
    except Exception as e:                                 # missing arrays or fields
        raise StaleIndex(f"incomplete index file ({type(e).__name__}: {e})") from None


def _rebuild(meta: dict, a: Dict[str, np.ndarray], source: Source) -> StoredIndex:
    size = source.size
    kind = a["chunks/kind"]
    n_chunks = len(kind)
    cols = ("chunks/start", "chunks/records_start", "chunks/compressed_size", "chunks/uncompressed_size",
            "chunks/compression")
    if any(len(a[c]) != n_chunks for c in cols):
        raise StaleIndex("chunk table columns differ in length")
    chunks: List = []
    for k in range(n_chunks):
        start, rs = int(a["chunks/start"][k]), int(a["chunks/records_start"][k])
        csize, usize = int(a["chunks/compressed_size"][k]), int(a["chunks/uncompressed_size"][k])
        comp = str(a["chunks/compression"][k])
        if kind[k] == 1:
            if usize > size:
                raise StaleIndex("messages outside chunks reach past the end of the recording")
            chunks.append(_LooseMessages(usize))
        else:
            if rs + csize > size or comp not in ("", "zstd", "lz4"):
                raise StaleIndex(f"chunk {k} lies outside the recording")
            chunks.append(StoredChunk(start, rs, comp, csize, usize))

    schemas = {s["id"]: Schema(id=s["id"], name=s["name"], encoding=s["encoding"],
                               data=base64.b64decode(s["data_b64"])) for s in meta["schemas"]}
    channels = {c["id"]: Channel(id=c["id"], topic=c["topic"], schema_id=c["schema_id"],
                                 message_encoding=c["message_encoding"], metadata=c["metadata"])
                for c in meta["channels"]}
    cm = meta["container"]
    container = Container.from_stored(source, schemas, channels, chunks, bool(cm["recovered"]),
                                      int(cm["unchunked_messages"]), int(cm["ignored_tail_bytes"]),
                                      bool(cm["has_message_indexes"]))
    out = StoredIndex(container)
    for t in meta["topics"]:
        key, n = t["key"], int(t["frames"])
        if t["channel_id"] not in channels:
            raise StaleIndex(f"{t['topic']}: unknown channel")
        chunk, offset, bound, log_time = (a[f"{key}/{c}"] for c in ("chunk", "offset", "bound", "log_time"))
        if not all(len(x) == n for x in (chunk, offset, bound, log_time)):
            raise StaleIndex(f"{t['topic']}: frame arrays differ in length")
        if n and (int(chunk.max()) >= n_chunks):
            raise StaleIndex(f"{t['topic']}: a frame refers to a chunk that is not in the index")
        limits = np.array([c.uncompressed_size for c in chunks], np.uint64)
        if n and ((offset >= bound).any() or (bound > limits[chunk]).any()):
            raise StaleIndex(f"{t['topic']}: a frame lies outside its chunk")
        refs = [MessageRef(int(c), int(o), int(b), int(lt)) for c, o, b, lt in zip(chunk, offset, bound, log_time)]
        index = FrameIndex(refs, log_time.astype(np.int64))
        scan = _rebuild_scan(t, a, n) if t.get("scanned") else None
        out.topics[t["topic"]] = StoredTopic(t["topic"], t["channel_id"], t.get("codec"), index, scan)
    return out


def _rebuild_scan(t: dict, a: Dict[str, np.ndarray], n: int) -> scanmod.ScanResult:
    key = t["key"]
    stamp, flags, decodable, params = (a[f"{key}/{c}"] for c in ("stamp", "flags", "decodable", "params"))
    data, offs, sizes = a[f"{key}/param_data"], a[f"{key}/param_offset"], a[f"{key}/param_size"]
    if not all(len(x) == n for x in (stamp, flags, decodable, params)) or len(offs) != len(sizes):
        raise StaleIndex(f"{t['topic']}: scan arrays differ in length")
    if not np.isin(decodable, (scanmod.DECODABLE, scanmod.NOT_DECODABLE, scanmod.UNKNOWN)).all():
        raise StaleIndex(f"{t['topic']}: unknown decodability values")
    if n and (int(params.max()) >= len(sizes) or int(params.min()) < -1):
        raise StaleIndex(f"{t['topic']}: a frame refers to parameter sets that are not in the index")
    if len(sizes) and int((offs + sizes).max()) > len(data):
        raise StaleIndex(f"{t['topic']}: parameter sets reach past their data")
    raw = data.tobytes()
    param_sets = [raw[int(o):int(o) + int(s)] for o, s in zip(offs, sizes)]
    return scanmod.ScanResult(
        t["codec"], (flags & KEYFRAME).astype(bool), (flags & CARRIES_PARAMS).astype(bool),
        decodable.astype(np.int8), int(t["b_frames"]), stamp.astype(np.int64), t["gaps_allowed"],
        t["params_first_seen"], [tuple(g) for g in t["reference_gaps"]], params.astype(np.int32), param_sets)
