# mcap-frames

Frame-accurate video access for MCAP files (community project, not affiliated with Foxglove).

Frame *n* is always the *n*-th message of the channel, in the order the messages were written (the order a
decoder needs). Getting it decodes from the nearest keyframe and reads only the chunks (or, when chunks are
uncompressed, the byte ranges) that hold those frames, using the file's own `ChunkIndex` / `MessageIndex`
records; nothing has to be stored next to the file
([index files](https://github.com/recording-store/mcap-frames/blob/main/docs/index-file.md) are optional). Frame numbers
never depend on what a decoder manages to output, so they do not shift when a stream is damaged: for H.264, a
frame that depends on a missing picture raises `FrameNotDecodable` (inferred from `frame_num`, without decoding).

```bash
pip install "mcap-frames[decode]"     # [decode] adds PyAV and Pillow; indexing and checks work without it
```

```python
import mcap_frames as mf

rec = mf.open("episode.mcap")                 # or an http(s) URL that supports range requests
# rec = mf.open(mf.HttpSource(url, headers={...}, read_timeout=30, retries=3))
print(rec.video_topics)
v = rec.video("/camera/front")

f = v.frame(120)                              # f.image: RGB array; f.log_time_ns, f.stamp_ns, f.decoded_from
frames = v.frames([10, 500, 900])             # frames sharing a keyframe are decoded once
for f in v.iter_frames(range(len(v)), on_error="skip"):   # streamed; undecodable frames left out
    ...
run = v.encoded(120)                          # raw access units from the keyframe, for your own decoder
n = v.index_at(time_ns, mode="nearest")       # log_time → frame number
n = v.index_at(time_ns, clock="capture")      # by the capture timestamp carried in the messages
v.keyframes, v.stamps_ns, v.decodable(600)    # need one pass over the headers (rec.scan() does all topics)

report = mf.check("episode.mcap")             # health check
print(report.summary())

mf.export_frames(v, [10, 500], "out/")        # images + frames.json with a status per frame
mf.to_mp4(v, "clip.mp4", 300, 900)            # stream copy, no re-encoding

rec.save_index("episode.mcap.frames")         # where every frame is + what a scan finds (no verdicts)
rec = mf.open("episode.mcap", index="episode.mcap.frames")   # next time: reads only the end of the recording
```

```bash
mcap-frames info  episode.mcap
mcap-frames check recordings/ --json report.json      # exit code 1 when a file has errors
mcap-frames frames episode.mcap --topic /camera/front --index 10 500 --out out/
mcap-frames mp4   episode.mcap --topic /camera/front --start-index 300 --end-index 900 --out clip.mp4
mcap-frames index recordings/ --out-dir idx/          # index files; --no-scan: locations only; --check report.json
mcap-frames check recordings/ --index-dir idx/        # health check from the index files
```

## Index files

Opening a recording finds where its frames are, and a scan reads every frame's first bytes (keyframes, parameter
sets, capture times, `frame_num`). Both can be stored in an index file, so that the next open reads the index file
and the last 64 KiB of the recording, and nothing else: health checks, keyframe lists and lookups by capture time
then need no further reading, which matters most on object storage. Index files hold facts, not health-check
verdicts; an index that does not match its recording is ignored (`rec.index_status` says why). Layout and rules:
[docs/index-file.md](https://github.com/recording-store/mcap-frames/blob/main/docs/index-file.md).

## What the health check reports

Each issue says what it is based on (`evidence`) and whether it is `inferred`. Most are facts read from the
bitstream or the file. The timing issues are conclusions about what happened while recording, drawn from
timestamps: they are only as right as the timestamps, so they come with how regular the timestamps are.

| code | level | based on | meaning |
|---|---|---|---|
| `missing-reference` | error | frame_num | frames after the first keyframe depend on a missing frame; tools that count decoded pictures shift after them. For H.264 it also says what went missing (reference frames, or a keyframe) |
| `b-frames` | error | slice headers | message order differs from picture order, so frames cannot be addressed by number |
| `no-keyframe`, `params-missing` | error | NAL units | nothing can be decoded |
| `timestamp-backwards` | error | log_time | log_time goes backwards |
| `starts-mid-gop` | warning | NAL units | the first frames come before the first keyframe |
| `frames-skipped` | warning | timestamps, **inferred** | whole frames are missing from the timeline (skipped by the camera, or lost before reaching the file): frame numbers are fine, but time and frame number no longer advance together |
| `timestamp-not-increasing` | warning | capture timestamps | capture timestamps repeat or go backwards |
| `truncated` | warning | file structure | the recording was cut off; complete chunks and messages were recovered |
| `params-not-repeated` | info | NAL units | keyframes rely on parameter sets sent earlier |
| `no-message-index`, `unchunked` | info | file structure | the file had to be read once when opened to locate frames |
| `timestamp-jitter` | info | timestamps, **inferred** | long intervals that nearby frames make up (frames stamped late or early); no frame is missing |
| `timing-irregular` | info | timestamps | frames arrive in bursts, so timestamps cannot show skipped frames |
| `irregular-gop` | info | NAL units | keyframe interval varies (scene-cut keyframes, or keyframes that were lost) |
| `decodability-unknown` | info | slice headers | cannot be inferred (H.265, or H.264 streams that allow frame_num gaps) |

Which H.264 frames can be decoded is worked out from `frame_num` in the slice headers, without decoding.

Timing is checked on the capture timestamp carried in each message when every frame has one, otherwise on
log_time. Frames are compared with a fixed-rate grid whose period is measured over the whole topic: a missing
frame shifts all later frames by a whole period, while a late or early frame is off the grid on its own. A shift
of about half a period is a clock slip, not a missing frame. When most intervals are far from the period (frames
delivered in bursts), timing is not analysed.

## Supported messages

`foxglove.CompressedVideo`, `foxglove.CompressedImage` (protobuf); `foxglove_msgs/msg/CompressedVideo`,
`foxglove_msgs/msg/CompressedImage`, `sensor_msgs/msg/CompressedImage` (ROS 2). H.264 and H.265 video, JPEG /
PNG / WebP images. Chunks may be uncompressed, zstd or lz4.

Files without the usual indexes are read once when opened, then accessed like any other file:

- no `MessageIndex` records: chunks are read to locate the messages;
- messages written outside chunks (for example ROS 2's `fastwrite` preset): the file is walked once;
- recordings that were cut off (no summary): complete chunks and messages are recovered, the unfinished record
  at the end is dropped.

## Verifying it yourself

`verify/verify.py` decodes every message of a topic in order, tagging each packet with its message number, so
every picture a decoder puts out can be traced to its message even when the decoder drops or conceals frames. It
then checks that every frame mcap-frames says can be decoded is pixel-identical to that picture, and that every
frame it says cannot be decoded raises an error. It also counts the frames a reader that numbers frames by
counting decoded pictures would get wrong.

```bash
pip install -e ".[decode]"
python verify/verify.py synthetic                 # recordings with known damage, ~5 s
huggingface-cli download Voxel51/ABC-130k --repo-type dataset --include "data/**" --local-dir abc130k
python verify/verify.py dataset abc130k/data      # 40 episodes, 3.4 GB, ~15 min; --all checks every frame
```

## Status and limitations

0.1.0 is the first release; the API may still change before 1.0
([changelog](https://github.com/recording-store/mcap-frames/blob/main/CHANGELOG.md)). Verified with
`verify/verify.py` on synthetic streams, including damaged ones, and on the public
[ABC-130k](https://huggingface.co/datasets/Voxel51/ABC-130k) subset.

- Missing references are inferred for H.264 only. For H.265, and for H.264 streams that allow `frame_num`
  gaps, a damaged frame is only detected when the decoder drops it; a decoder that conceals the damage
  returns a picture.
- Streams with B-frames are refused (message order differs from picture order), for H.264 and H.265.
- ROS `compressedDepth` images (a header before the PNG) are not supported.
- When a topic has several channels, only the first one is used.
- HTTP sources reuse connections, retry transient failures (connection errors, timeouts, 408 / 429 / 5xx) and
  follow redirects. Authentication is limited to presigned URLs or headers you pass in; proxies from the
  environment are not used. Tested against an S3-compatible object store with presigned GET URLs.
- Without an index file, the index is rebuilt each time a file is opened.
- `frames()` keeps every picture in memory; `iter_frames()` keeps about `workers` runs (keyframe to frame)
  of pictures, so lower `workers` to use less memory.
- Pictures come back as RGB at full size; convert or resize them yourself.
- A `Recording` and its streams are not thread-safe: open the file once per thread.
- Interlaced (field-coded) H.264 has not been tested.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Tests generate their own recordings (`tests/fixtures.py`): every picture carries its source frame number as
black / white blocks, so any decode can be traced back to the frame it came from.

License: Apache-2.0.
