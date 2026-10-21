# Index files

Added in 0.1.0 (`mcap_frames/indexfile.py`).

Opening a recording builds a frame index in memory: where every frame of every video topic is, and, after a
scan, which frames are keyframes, which can be decoded and when each was captured. Building it costs one small
read per chunk, and the scan reads every chunk (compressed chunks whole). On an object store that is seconds to
open and minutes to scan a multi-gigabyte file, every time it is opened. Files without indexes (no MessageIndex
records, messages outside chunks, truncated files) have to be walked whole each time.

An index file stores what was built so that the next open reads the index file and the end of the recording, and
nothing else.

## Three layers: locating, scanning, checking

| layer | what it does | result | in the index file |
|---|---|---|---|
| locate | where every frame is in the file | facts | always |
| scan | the first bytes of every frame: keyframe or not, where the parameter sets are, capture timestamp, whether `frame_num` runs on without a gap | facts, no verdict | by default |
| check | the health-check rules turn those facts into errors, warnings and notes | verdicts | never |

Indexing never runs the health-check rules. Callers with their own quality checks index without ours and read the
stored facts (keyframes, capture times, decodability) or ignore them; callers who want ours ask for it, and it
costs no further read of the recording, since the scan already holds what the rules need.

Correct frame access does not depend on the scan: without one, getting frame n still walks from the nearest
keyframe and checks `frame_num` on the way, and raises if a reference is missing. What a stored scan saves is
reading the recording again: keyframe lists, lookups by capture time, health checks and the walk back to a
keyframe all come from the index file.

The scan is done by default because an index is usually written once, where the data lands, which is where reading
the whole file is cheapest; every later open, often remote, then saves that read. `scan=False` / `--no-scan`
writes a locating-only index, the fastest to build (only index records are read); its topics can be scanned later
and the index written again.

## Behaviour

- One index file describes one MCAP file and any number of its topics.
- Index files are only read from and written to paths the caller names; the library never looks for one or writes
  one on its own.
- An index file that does not match the recording (or is damaged, or of another format version) is ignored: the
  index is built from the recording as if there were none, and `Recording.index_status` says so.
- Index files hold facts read from the file and the bitstream, never health-check conclusions: the checks are
  recomputed from these facts, so a report never comes from older rules.

```python
rec = mf.open("episode.mcap", index="episode.mcap.frames")   # used when valid, else built in memory
rec.index_status                                              # "loaded", "missing", "stale: <reason>"
rec.save_index("episode.mcap.frames")                         # locate + scan every video topic
rec.save_index("episode.mcap.frames", scan=False)             # locate only
mf.check(rec)                                                 # our rules; from the stored scan when there is one
mf.open(url, index=url + ".frames")                           # an index next to a remote object (read only)
```

```bash
mcap-frames index recordings/ --out-dir idx/                      # locate + scan; no verdicts
mcap-frames index recordings/ --out-dir idx/ --no-scan            # locate only
mcap-frames index recordings/ --out-dir idx/ --check report.json  # also our health check, from the same pass
mcap-frames check recordings/ --index-dir idx/                    # health check from stored indexes
```

`index` writes one `<name>.mcap.frames` per recording into `--out-dir`; for a directory given, its layout is
mirrored there, so recordings with the same name in different directories keep apart. With `--index-dir`, `check`
(and `info`, `frames`, `mp4`) look for the index at the same place and use it when it matches the recording.

Writes are atomic (a temporary file in the same directory, then renamed). Remote locations are read only.

## Matching an index to a recording

Stored when the index is written, compared when it is loaded; any difference makes the index stale:

| field | how it is obtained | why |
|---|---|---|
| `size` | the source's size | changes with almost every edit |
| `tail_sha256` | sha256 of the last min(size, 64 KiB) bytes | covers the footer and summary of a finished file, and the end of a truncated one; one read |
| `etag` | HTTP sources, when both sides have one | an object store's own content version |
| `sha256` | only when the caller passes it (`index_sha256=`) | for callers that already know the file's hash |

Opening with a valid index therefore costs one read of the index file and one read of the recording's last
64 KiB (over HTTP, plus the request that gets the recording's size).

A change that keeps the size and the last 64 KiB of a recording is not noticed. Recordings are not edited in
place, so this is a fair trade for not reading the whole file; callers who know the file's hash pass `sha256` to
`save_index` and `mf.open`, and an index made for another hash is then stale.

## File layout

A NumPy `.npz` archive (zip of `.npy` arrays; written with `numpy.savez_compressed`, read with
`allow_pickle=False`). `.npy` is a documented format, so other languages can read it with a zip reader and a
small header parser. All integers little-endian.

### `meta` (0-d unicode array holding JSON)

```json
{
  "format": "mcap-frames-index",
  "version": 1,
  "created_by": "mcap-frames 0.1.0",
  "file": {"size": 1073741824, "tail_sha256": "…", "etag": "\"…\"", "sha256": null},
  "container": {"recovered": false, "unchunked_messages": 0, "ignored_tail_bytes": 0},
  "schemas": [{"id": 1, "name": "sensor_msgs/msg/CompressedImage", "encoding": "ros2msg", "data_b64": "…"}],
  "channels": [{"id": 3, "topic": "/camera/front", "schema_id": 1, "message_encoding": "cdr", "metadata": {}}],
  "topics": [{"key": "t0", "topic": "/camera/front", "channel_id": 3, "frames": 9000, "codec": "h264",
              "scanned": true, "b_frames": 0, "gaps_allowed": false, "params_first_seen": 0,
              "reference_gaps": [[120, 10, 1, 16]]}]
}
```

### Chunks

One row per chunk of the recording, in file order. Messages written outside chunks are one pseudo-chunk with
`kind` 1, whose offsets are file offsets.

| array | dtype | content |
|---|---|---|
| `chunks/start` | uint64 | file offset of the chunk record (0 for the pseudo-chunk) |
| `chunks/records_start` | uint64 | file offset of the chunk's records |
| `chunks/compressed_size`, `chunks/uncompressed_size` | uint64 | sizes of the records |
| `chunks/compression` | unicode | `""`, `"zstd"` or `"lz4"` |
| `chunks/kind` | uint8 | 0 chunk, 1 messages outside chunks |

### Per topic (`<key>/…`, one row per frame, in write order = decode order)

Always present:

| array | dtype | content |
|---|---|---|
| `chunk` | uint32 | row in the chunk table |
| `offset` | uint64 | the message record's offset in the chunk's uncompressed records |
| `bound` | uint64 | upper bound of the record (where the next record starts) |
| `log_time` | uint64 | nanoseconds |

Present when `scanned` is true:

| array | dtype | content |
|---|---|---|
| `stamp` | int64 | capture timestamp carried in the message; -1 when it has none |
| `flags` | uint8 | bit 0 keyframe, bit 1 carries its parameter sets |
| `decodable` | int8 | 1 yes, 0 no, -1 cannot tell |
| `params` | int32 | parameter sets to put before this frame when decoding starts here: row in `param_*`, -1 none |
| `param_data` / `param_offset` / `param_size` | uint8 / uint64 / uint64 | distinct parameter sets (SPS + PPS, and VPS for H.265), as bytes |

About 37 bytes a frame before compression; 16 bytes a frame as written on the public ABC-130k subset
(316,835 frames in 5.1 MB of index files).

## Validation on load

Before an index is used: format and version match; `size`, `tail_sha256` (and `etag`, `sha256` when present)
match; every per-topic array has the topic's `frames` rows; every `chunk` value is a row of the chunk table;
offsets and sizes fall inside the file. Anything else is treated as stale.

## Tests

- Round trip: for every synthetic case in `verify/verify.py`, the loaded index equals the built one (frame
  locations, scan arrays) and health reports are identical.
- Stale detection: a changed byte at the end, a different size, a truncated index file, another version, arrays of
  the wrong length, a chunk row out of range.
- Files without indexes, unchunked and truncated: opened from an index without walking the file.
- HTTP: opening with a valid index makes two requests: the index file, and the recording's last 64 KiB.
- A health check on a recording opened from a scanned index reads nothing more than that tail, and its report
  equals the one from the recording itself; `index --check` writes the same report as `check`.
- A locating-only index: frames still come back right (and undecodable ones still raise); scanning later and saving
  again gives the same file as a scanned index built in one go.
- `verify/verify.py synthetic --index` and `dataset DIR --index` give the same results as without index files.
