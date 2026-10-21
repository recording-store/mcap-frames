# Changelog

## 0.1.0

First release.

- Frame access by number: frame *n* is the *n*-th message of a video topic, decoded from the nearest keyframe,
  reading only the chunks (or byte ranges) that hold the frames. `frame`, `frames`, `iter_frames` (streamed,
  `on_error="skip"`), `encoded` (raw access units for your own decoder), `index_at` by log time or capture time.
- H.264 frames that depend on a missing picture raise `FrameNotDecodable`, inferred from `frame_num` without
  decoding, so frame numbers never shift on damaged streams. Streams with B-frames are refused (H.264, H.265).
- Health check (`mf.check`, `mcap-frames check`): missing references, B-frames, missing keyframes and parameter
  sets, timestamp problems, skipped frames told apart from jitter; every issue says what it is based on and
  whether it is inferred.
- Export: images with a status per frame (`export_frames`), MP4 by stream copy (`to_mp4`).
- Index files (`save_index`, `mf.open(..., index=...)`, `mcap-frames index`): where every frame is and what a
  scan found, so the next open reads only the index file and the end of the recording. See
  [docs/index-file.md](docs/index-file.md).
- Sources: local files and http(s) URLs with range requests (persistent connections, retries, timeouts).
- Files without the usual indexes: no `MessageIndex` records, messages written outside chunks, recordings that
  were cut off.
- `verify/verify.py` checks every frame against a sequential decode, on synthetic recordings or a dataset.
