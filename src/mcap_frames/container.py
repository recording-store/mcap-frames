"""The MCAP layer: summary section, chunk indexes, message indexes, and reading single messages.

Only the file's own index records are used, so opening a file costs one read for the footer, one for the
summary section, and one per chunk for its message indexes. Message bytes are read on demand:

- uncompressed chunk: only the bytes of the requested messages (nearby messages merged into one read);
- compressed chunk (zstd / lz4): the whole chunk is read and decompressed once, and kept in a small cache.
"""

from __future__ import annotations

import io
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mcap.exceptions import EndOfFile
from mcap.records import Channel, ChunkIndex, Footer, Schema, Statistics
from mcap.stream_reader import StreamReader

from mcap_frames.errors import UnsupportedFile
from mcap_frames.source import Source

MAGIC = b"\x89MCAP0\r\n"
FOOTER_SIZE = 1 + 8 + 8 + 8 + 4                 # opcode, record length, summary_start, summary_offset_start, crc
MESSAGE_HEADER = 1 + 8 + 2 + 4 + 8 + 8          # opcode, record length, channel, sequence, log_time, publish_time
OP_FOOTER, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_CHUNK, OP_MESSAGE_INDEX, OP_DATA_END = 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0F


@dataclass
class MessageRef:
    chunk: int                                   # position in Container.chunks
    offset: int                                  # offset of the message record in the chunk's uncompressed records
    bound: int                                   # where the next indexed record starts (upper bound of this record)
    log_time: int


class Container:
    def __init__(self, source: Source, chunk_cache: int = 4):
        self.source = source
        if source.size < len(MAGIC) * 2 + FOOTER_SIZE:
            raise UnsupportedFile(f"{source.name} is too small to be an MCAP file")
        self.schemas: Dict[int, Schema] = {}
        self.channels: Dict[int, Channel] = {}
        self.chunks: List[ChunkIndex] = []
        self.statistics: Optional[Statistics] = None
        self.recovered = False             # True: no usable summary (truncated file); chunks found by reading
        self.ignored_tail_bytes = 0        # bytes at the end that did not form a complete record
        self._message_indexes: Dict[int, Dict[int, np.ndarray]] = {}
        self._chunk_cache: "OrderedDict[int, bytes]" = OrderedDict()
        self._chunk_cache_size = chunk_cache

        tail = source.read(source.size - FOOTER_SIZE - len(MAGIC), FOOTER_SIZE + len(MAGIC))
        summary_start = struct.unpack_from("<Q", tail, 9)[0] if tail[-len(MAGIC):] == MAGIC else 0
        if summary_start == 0:
            self._recover()
            return
        summary = source.read(summary_start, source.size - len(MAGIC) - summary_start)   # summary + footer
        for rec in _records(summary):
            if isinstance(rec, Schema):
                self.schemas[rec.id] = rec
            elif isinstance(rec, Channel):
                self.channels[rec.id] = rec
            elif isinstance(rec, ChunkIndex):
                self.chunks.append(rec)
            elif isinstance(rec, Statistics):
                self.statistics = rec
        if not self.chunks and (self.statistics is None or self.statistics.message_count):
            raise UnsupportedFile(f"{source.name} has no chunk index; it cannot be read by index")
        self.chunks.sort(key=lambda c: c.chunk_start_offset)

    def _recover(self) -> None:
        """No summary (the recording was cut off, or written without one): walk the top-level records from the
        start, keep every complete chunk, and locate messages by reading the chunks. Reads the whole file."""
        src = self.source
        buf_start, buf = 0, b""

        def get(offset: int, n: int) -> bytes:              # buffered: one read per ~1 MB of record headers
            nonlocal buf_start, buf
            if not (buf_start <= offset and offset + n <= buf_start + len(buf)):
                buf_start = offset
                buf = src.read(offset, min(max(n, 1 << 20), src.size - offset))
            return buf[offset - buf_start:offset - buf_start + n]

        if get(0, len(MAGIC)) != MAGIC:
            raise UnsupportedFile(f"{src.name} is not an MCAP file")
        pos, op = len(MAGIC), None
        while pos + 9 <= src.size:
            op, length = struct.unpack("<BQ", get(pos, 9))
            end = pos + 9 + length
            if end > src.size:
                break                                     # the record the recorder was writing when it stopped
            if op == OP_CHUNK:
                start, stop, usize, _crc, clen = struct.unpack("<QQQII", get(pos + 9, 32))
                compression = get(pos + 41, clen).decode()
                rlen = struct.unpack("<Q", get(pos + 41 + clen, 8))[0]
                self.chunks.append(ChunkIndex(message_start_time=start, message_end_time=stop, chunk_start_offset=pos,
                                              chunk_length=9 + length, message_index_offsets={}, message_index_length=0,
                                              compression=compression, compressed_size=rlen, uncompressed_size=usize))
            elif op in (OP_SCHEMA, OP_CHANNEL):
                self._definition(get(pos, 9 + length))
            elif op in (OP_DATA_END, OP_FOOTER):
                break
            pos = end
        self.ignored_tail_bytes = src.size - pos if pos < src.size and op not in (OP_DATA_END, OP_FOOTER) else 0
        self.recovered = True
        self._index_by_reading_chunks()

    def _definition(self, record: bytes) -> None:
        for rec in _records(record):
            if isinstance(rec, Schema):
                self.schemas.setdefault(rec.id, rec)
            elif isinstance(rec, Channel):
                self.channels.setdefault(rec.id, rec)

    # ------------------------------------------------------------------ index records
    @property
    def has_message_indexes(self) -> bool:
        """False when the writer left out MessageIndex records (chunks must then be read to find messages)."""
        return any(c.message_index_offsets for c in self.chunks)

    def _index_by_reading_chunks(self) -> None:
        """Fallback for files without MessageIndex records: read every chunk once and locate its messages."""
        for k, c in enumerate(self.chunks):
            if k in self._message_indexes:
                continue
            if c.compression:
                data = _decompress(c.compression, self.source.read(self._records_start(k), c.compressed_size),
                                   c.uncompressed_size)
            else:
                data = self.source.read(self._records_start(k), c.compressed_size)
            per: Dict[int, List[Tuple[int, int]]] = {}
            pos, n = 0, len(data)
            while pos + 9 <= n:
                op, length = data[pos], struct.unpack_from("<Q", data, pos + 1)[0]
                if op in (OP_SCHEMA, OP_CHANNEL) and self.recovered:
                    self._definition(data[pos:pos + 9 + length])     # definitions often live inside chunks
                elif op == OP_MESSAGE:
                    channel = struct.unpack_from("<H", data, pos + 9)[0]
                    log_time = struct.unpack_from("<Q", data, pos + 9 + 2 + 4)[0]
                    per.setdefault(channel, []).append((log_time, pos))
                pos += 9 + length
            self._message_indexes[k] = {ch: np.array(v, dtype=np.int64) for ch, v in per.items()}

    def chunks_with(self, channel_id: int) -> List[int]:
        if self.recovered or not self.has_message_indexes:
            self._index_by_reading_chunks()
            return [k for k in range(len(self.chunks)) if channel_id in self._message_indexes[k]]
        return [i for i, c in enumerate(self.chunks) if channel_id in c.message_index_offsets]

    def message_indexes(self, chunk_ids: Sequence[int]) -> Dict[int, Dict[int, np.ndarray]]:
        """{chunk: {channel: array of (log_time, offset)}} for every channel in each chunk (one read per chunk)."""
        missing = [k for k in chunk_ids if k not in self._message_indexes]
        ranges = []
        for k in missing:
            c = self.chunks[k]
            start = min(c.message_index_offsets.values())
            ranges.append((start, c.message_index_length))
        for k, blob in zip(missing, self.source.read_many(ranges, merge_gap=0)):
            self._message_indexes[k] = _parse_message_indexes(blob)
        return {k: self._message_indexes[k] for k in chunk_ids}

    def refs(self, channel_id: int) -> List[MessageRef]:
        """Every message of a channel, in the order it was written (= decode order), whatever its log_time."""
        chunk_ids = self.chunks_with(channel_id)
        out: List[MessageRef] = []
        for k, per_channel in self.message_indexes(chunk_ids).items():
            mine = per_channel.get(channel_id)
            if mine is None or not len(mine):
                continue
            starts = np.unique(np.concatenate([a[:, 1] for a in per_channel.values()]))
            end = self.chunks[k].uncompressed_size
            nxt = np.searchsorted(starts, mine[:, 1], side="right")
            bounds = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)], end)
            out.extend(MessageRef(k, int(o), int(b), int(t)) for (t, o), b in zip(mine, bounds))
        out.sort(key=lambda r: (self.chunks[r.chunk].chunk_start_offset, r.offset))
        return out

    # ------------------------------------------------------------------ message bytes
    def _records_start(self, k: int) -> int:
        c = self.chunks[k]
        return c.chunk_start_offset + 1 + 8 + 8 + 8 + 8 + 4 + 4 + len(c.compression) + 8

    def _chunk_records(self, k: int) -> bytes:
        if k in self._chunk_cache:
            self._chunk_cache.move_to_end(k)
            return self._chunk_cache[k]
        c = self.chunks[k]
        raw = self.source.read(self._records_start(k), c.compressed_size)
        data = _decompress(c.compression, raw, c.uncompressed_size)
        self._chunk_cache[k] = data
        while len(self._chunk_cache) > self._chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return data

    def read_messages(self, refs: Sequence[MessageRef]) -> List[bytes]:
        """Message payloads (the serialized message, without the record header), in the order given."""
        out: List[Optional[bytes]] = [None] * len(refs)
        plain: List[int] = []
        for i, r in enumerate(refs):
            if self.chunks[r.chunk].compression:
                out[i] = _message_data(self._chunk_records(r.chunk), r.offset)
            else:
                plain.append(i)
        if plain:
            ranges = [(self._records_start(refs[i].chunk) + refs[i].offset, refs[i].bound - refs[i].offset) for i in plain]
            for i, blob in zip(plain, self.source.read_many(ranges)):
                out[i] = _message_data(blob, 0)
        return out                                        # type: ignore[return-value]

    def read_message_heads(self, refs: Sequence[MessageRef], head: int) -> List[Tuple[bytes, bool]]:
        """Like read_messages, but only the first ``head`` bytes of each message are read from uncompressed
        chunks. Returns (payload bytes, truncated) pairs. Compressed chunks still have to be read whole."""
        out: List[Optional[Tuple[bytes, bool]]] = [None] * len(refs)
        plain: List[int] = []
        for i, r in enumerate(refs):
            if self.chunks[r.chunk].compression:
                data = _message_data(self._chunk_records(r.chunk), r.offset)
                out[i] = (data[:head], len(data) > head)          # keep only the head: less memory
            else:
                plain.append(i)
        if plain:
            ranges = [(self._records_start(refs[i].chunk) + refs[i].offset,
                       min(refs[i].bound - refs[i].offset, MESSAGE_HEADER + head)) for i in plain]
            # heads are a few KB: merging across large gaps would read the picture data in between
            gap = min(self.source.merge_gap, 4 * head)
            for i, blob in zip(plain, self.source.read_many(ranges, merge_gap=gap)):
                length = struct.unpack_from("<Q", blob, 1)[0]
                end = 9 + length
                out[i] = (bytes(blob[MESSAGE_HEADER:min(end, len(blob))]), end > len(blob))
        return out                                        # type: ignore[return-value]

    def compression_of(self, refs: Sequence[MessageRef]) -> List[str]:
        return sorted({self.chunks[r.chunk].compression or "none" for r in refs})


def _records(blob: bytes):
    reader = StreamReader(io.BytesIO(blob), skip_magic=True)
    try:
        for rec in reader.records:
            yield rec
            if isinstance(rec, Footer):
                return
    except EndOfFile:
        return


def _parse_message_indexes(blob: bytes) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    pos = 0
    while pos + 9 <= len(blob):
        op, length = blob[pos], struct.unpack_from("<Q", blob, pos + 1)[0]
        body = pos + 9
        if op == OP_MESSAGE_INDEX:
            channel, n_bytes = struct.unpack_from("<HI", blob, body)
            pairs = np.frombuffer(blob, dtype="<u8", count=n_bytes // 8, offset=body + 6).reshape(-1, 2)
            out[channel] = pairs.astype(np.int64)
        pos = body + length
    return out


def _message_data(buf: bytes, offset: int) -> bytes:
    if buf[offset] != OP_MESSAGE:
        raise UnsupportedFile(f"expected a message record at offset {offset}, found opcode {buf[offset]:#x}")
    length = struct.unpack_from("<Q", buf, offset + 1)[0]
    return bytes(buf[offset + MESSAGE_HEADER:offset + 9 + length])


def _decompress(compression: str, raw: bytes, size: int) -> bytes:
    if compression == "":
        return raw
    if compression == "zstd":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(raw, max_output_size=size)
    if compression == "lz4":
        import lz4.frame
        return lz4.frame.decompress(raw)
    raise UnsupportedFile(f"unsupported chunk compression {compression!r}")


def describe_chunks(container: Container, chunk_ids: Sequence[int]) -> Tuple[int, int]:
    """(compressed bytes, uncompressed bytes) of the given chunks."""
    return (sum(container.chunks[k].compressed_size for k in chunk_ids),
            sum(container.chunks[k].uncompressed_size for k in chunk_ids))
