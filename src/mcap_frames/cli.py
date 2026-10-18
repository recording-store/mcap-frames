"""Command line: ``mcap-frames info | check | frames | mp4``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mcap_frames as mf


def _files(paths):
    out = []
    for p in paths:
        if p.startswith(("http://", "https://")):
            out.append(p)
        elif Path(p).is_dir():
            out += [str(x) for x in sorted(Path(p).rglob("*.mcap")) if not x.name.startswith("._")]
        else:
            out.append(p)
    return out


def cmd_info(a) -> int:
    with mf.open(a.file) as rec:
        for topic in rec.video_topics:
            v = rec.video(topic)
            t = v.log_times_ns
            dur = (int(t[-1]) - int(t[0])) / 1e9 if len(t) > 1 else 0
            print(f"{topic}\t{v.schema.name}\t{v.codec}\t{len(v)} frames\t{dur:.1f} s")
    return 0


def cmd_check(a) -> int:
    reports, ok = [], True
    for f in _files(a.paths):
        try:
            rep = mf.check(f, verify_decode=a.verify_decode)
        except mf.McapFramesError as e:
            print(f"{f}: cannot check: {e}", file=sys.stderr)
            ok = False
            continue
        ok = ok and rep.ok
        print(rep.summary())
        reports.append(rep.to_dict())
    if a.json:
        Path(a.json).write_text(json.dumps(reports, indent=1), encoding="utf-8")
    return 0 if ok else 1


def cmd_frames(a) -> int:
    with mf.open(a.file) as rec:
        v = rec.video(a.topic)
        frames = list(a.index or []) + [v.index_at(int(t)) for t in (a.time or [])]
        rows = mf.export_frames(v, frames, a.out, fmt=a.format)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"wrote {len(rows) - len(bad)} frames to {a.out}" + (f"; {len(bad)} not decodable" if bad else ""))
    return 0 if not bad else 1


def cmd_mp4(a) -> int:
    with mf.open(a.file) as rec:
        info = mf.to_mp4(rec.video(a.topic), a.out, a.start_index, a.end_index, a.allow_undecodable)
    print(json.dumps(info))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mcap-frames", description="Frame-accurate video access for MCAP files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="list image / video topics")
    p.add_argument("file")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("check", help="health check; exit code 1 when a file has errors")
    p.add_argument("paths", nargs="+", help="MCAP files, directories, or http(s) URLs")
    p.add_argument("--json", help="also write the reports to this JSON file")
    p.add_argument("--verify-decode", action="store_true", help="decode streams with missing references to confirm")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("frames", help="write frames as images plus frames.json")
    p.add_argument("file")
    p.add_argument("--topic", required=True)
    p.add_argument("--index", type=int, nargs="+", help="frame numbers (message order)")
    p.add_argument("--time", nargs="+", help="log_time in nanoseconds; the nearest frame is taken")
    p.add_argument("--format", default="jpeg", choices=("jpeg", "png"))
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("mp4", help="copy a range of frames into an MP4 without re-encoding")
    p.add_argument("file")
    p.add_argument("--topic", required=True)
    p.add_argument("--start-index", type=int)
    p.add_argument("--end-index", type=int)
    p.add_argument("--allow-undecodable", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_mp4)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
