# mcap-frames

Frame-accurate video access for MCAP files (community project, not affiliated with Foxglove).

Frame *n* is always the *n*-th message of the channel, in the order the messages were written (the order a
decoder needs). Getting it decodes from the nearest keyframe and reads only the chunks (or, when chunks are
uncompressed, the byte ranges) that hold those frames, using the file's own `ChunkIndex` / `MessageIndex`
records; nothing is stored next to the file. Frame numbers never depend on what a decoder manages to output,
so they do not shift when a stream is damaged: for H.264, a frame that depends on a missing picture raises
`FrameNotDecodable` (inferred from `frame_num`, without decoding).

```bash
pip install "mcap-frames[decode]"     # [decode] adds PyAV and Pillow; indexing and checks work without it
```

```python
import mcap_frames as mf

rec = mf.open("episode.mcap")                 # or an http(s) URL that supports range requests
print(rec.video_topics)
v = rec.video("/camera/front")

f = v.frame(120)                              # f.image: RGB array; f.log_time_ns, f.stamp_ns, f.decoded_from
frames = v.frames([10, 500, 900])             # frames sharing a keyframe are decoded once
run = v.encoded(120)                          # raw access units from the keyframe, for your own decoder
n = v.index_at(time_ns, mode="nearest")       # log_time → frame number

report = mf.check("episode.mcap")             # health check
print(report.summary())

mf.export_frames(v, [10, 500], "out/")        # images + frames.json with a status per frame
mf.to_mp4(v, "clip.mp4", 300, 900)            # stream copy, no re-encoding
```

```bash
mcap-frames info  episode.mcap
mcap-frames check recordings/ --json report.json      # exit code 1 when a file has errors
mcap-frames frames episode.mcap --topic /camera/front --index 10 500 --out out/
mcap-frames mp4   episode.mcap --topic /camera/front --start-index 300 --end-index 900 --out clip.mp4
```

## What the health check reports

| code | level | meaning |
|---|---|---|
| `missing-reference` | error | frames after the first keyframe depend on a missing frame; tools that count decoded pictures shift after them |
| `b-frames` | error | message order differs from picture order, so frames cannot be addressed by number |
| `no-keyframe`, `params-missing` | error | nothing can be decoded |
| `timestamp-backwards` | error | log_time goes backwards |
| `starts-mid-gop` | warning | the first frames come before the first keyframe |
| `timestamp-gap` | warning | intervals longer than 1.5× the median (frames skipped by the camera) |
| `params-not-repeated` | info | keyframes rely on parameter sets sent earlier |
| `irregular-gop` | info | keyframe interval varies (often scene-cut keyframes; harmless) |
| `decodability-unknown` | info | cannot be inferred (H.265, or H.264 streams that allow frame_num gaps) |

Which H.264 frames can be decoded is inferred from `frame_num` in the slice headers, without decoding.

## Supported messages

`foxglove.CompressedVideo`, `foxglove.CompressedImage` (protobuf); `foxglove_msgs/msg/CompressedVideo`,
`foxglove_msgs/msg/CompressedImage`, `sensor_msgs/msg/CompressedImage` (ROS 2). H.264 and H.265 video, JPEG /
PNG / WebP images. Chunks may be uncompressed, zstd or lz4.

Files without `MessageIndex` records, and recordings that were cut off (no summary), are read chunk by chunk;
a truncated file is recovered from its complete chunks.

## Status and limitations

0.1 development version; the API may still change. Verified against sequential decoding (pixel-identical) on
synthetic streams, including damaged ones, and on recordings from the public
[ABC-130k](https://huggingface.co/datasets/Voxel51/ABC-130k) subset.

- Missing references are inferred for H.264 only. For H.265, and for H.264 streams that allow `frame_num`
  gaps, a damaged frame is only detected when the decoder drops it; a decoder that conceals the damage
  returns a picture.
- Streams with B-frames are refused (message order differs from picture order).
- When a topic has several channels, only the first one is used.
- HTTP sources are experimental: no connection reuse, retries or authentication yet.
- Indexes live in memory only; they are rebuilt each time a file is opened.
- Interlaced (field-coded) H.264 has not been tested.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Tests generate their own recordings (`tests/fixtures.py`): every picture carries its source frame number as
black / white blocks, so any decode can be traced back to the frame it came from.

License: Apache-2.0.
