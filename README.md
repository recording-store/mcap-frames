# mcap-frames

Frame-accurate video access for MCAP files (community project, not affiliated with Foxglove).

- Build a per-frame index from the MCAP file's own `ChunkIndex` / `MessageIndex` records, without storing sidecar files.
- Fetch frame *n* by message index or time, reading and decoding only from the preceding keyframe.
- Health check: keyframe cadence, frames before the first keyframe, missing keyframes and which frames cannot be decoded, parameter sets, timestamp gaps.
- Export frames or MP4 without silently shifting frame numbers when a stream is damaged.

Status: not yet started.
