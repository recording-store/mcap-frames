"""Command line: ``mcap-frames info | check | index | frames | mp4``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import mcap_frames as mf


def _files(paths):
    """(recording, name of its index file relative to an index directory) for every recording named."""
    out = []
    for p in paths:
        if p.startswith(("http://", "https://")):
            out.append((p, urlsplit(p).path.rsplit("/", 1)[-1] + ".frames"))
        elif Path(p).is_dir():
            out += [(str(x), str(x.relative_to(p)) + ".frames") for x in sorted(Path(p).rglob("*.mcap"))
                    if not x.name.startswith("._")]
        else:
            out.append((p, Path(p).name + ".frames"))
    return out


def _open(path: str, index_name: str, index_dir):
    return mf.open(path, index=str(Path(index_dir) / index_name) if index_dir else None)


def _open_one(a):
    return _open(a.file, _files([a.file])[0][1], a.index_dir)


def cmd_info(a) -> int:
    with _open_one(a) as rec:
        for topic in rec.video_topics:
            v = rec.video(topic)
            t = v.log_times_ns
            dur = (int(t[-1]) - int(t[0])) / 1e9 if len(t) > 1 else 0
            print(f"{topic}\t{v.schema.name}\t{v.codec}\t{len(v)} frames\t{dur:.1f} s")
    return 0


def cmd_check(a) -> int:
    reports, ok = [], True
    for f, name in _files(a.paths):
        try:
            with _open(f, name, a.index_dir) as rec:
                if a.index_dir and rec.index_status != "loaded":
                    print(f"{f}: index {rec.index_status}; reading the recording", file=sys.stderr)
                rep = mf.check(rec, verify_decode=a.verify_decode)
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


def cmd_index(a) -> int:
    reports, ok = [], True
    for f, name in _files(a.paths):
        out = Path(a.out_dir) / name
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with mf.open(f) as rec:
                rec.save_index(out, scan=not a.no_scan)
                frames = sum(len(rec.video(t)) for t in rec.video_topics)
                print(f"{f} -> {out}: {len(rec.video_topics)} topics, {frames} frames"
                      + ("" if a.no_scan else ", scanned"))
                if a.check:
                    rep = mf.check(rec)
                    ok = ok and rep.ok
                    print(rep.summary())
                    reports.append(rep.to_dict())
        except (mf.McapFramesError, OSError) as e:
            print(f"{f}: cannot index: {e}", file=sys.stderr)
            ok = False
    if a.check:
        Path(a.check).write_text(json.dumps(reports, indent=1), encoding="utf-8")
    return 0 if ok else 1


def cmd_frames(a) -> int:
    with _open_one(a) as rec:
        v = rec.video(a.topic)
        frames = list(a.index or []) + [v.index_at(int(t)) for t in (a.time or [])]
        rows = mf.export_frames(v, frames, a.out, fmt=a.format)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"wrote {len(rows) - len(bad)} frames to {a.out}" + (f"; {len(bad)} not decodable" if bad else ""))
    return 0 if not bad else 1


def cmd_mp4(a) -> int:
    with _open_one(a) as rec:
        info = mf.to_mp4(rec.video(a.topic), a.out, a.start_index, a.end_index, a.allow_undecodable)
    print(json.dumps(info))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mcap-frames", description="Frame-accurate video access for MCAP files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    index_dir = dict(help="use index files written by `mcap-frames index --out-dir` to this directory, when they "
                          "match the recording")
    p = sub.add_parser("info", help="list image / video topics")
    p.add_argument("file")
    p.add_argument("--index-dir", **index_dir)
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("check", help="health check; exit code 1 when a file has errors")
    p.add_argument("paths", nargs="+", help="MCAP files, directories, or http(s) URLs")
    p.add_argument("--json", help="also write the reports to this JSON file")
    p.add_argument("--verify-decode", action="store_true", help="decode streams with missing references to confirm")
    p.add_argument("--index-dir", **index_dir)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("index", help="write index files: where every frame is, and (by default) what a scan finds")
    p.add_argument("paths", nargs="+", help="MCAP files, directories, or http(s) URLs")
    p.add_argument("--out-dir", required=True, help="index files go here, as <name>.mcap.frames (directories "
                                                    "given are mirrored, so equal names do not collide)")
    p.add_argument("--no-scan", action="store_true", help="only where frames are: fastest, reads only index records")
    p.add_argument("--check", metavar="REPORT_JSON", help="also run the health check (no further reading) and write "
                                                          "the reports here; exit code 1 when a file has errors")
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("frames", help="write frames as images plus frames.json")
    p.add_argument("file")
    p.add_argument("--topic", required=True)
    p.add_argument("--index", type=int, nargs="+", help="frame numbers (message order)")
    p.add_argument("--time", nargs="+", help="log_time in nanoseconds; the nearest frame is taken")
    p.add_argument("--format", default="jpeg", choices=("jpeg", "png"))
    p.add_argument("--out", required=True)
    p.add_argument("--index-dir", **index_dir)
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("mp4", help="copy a range of frames into an MP4 without re-encoding")
    p.add_argument("file")
    p.add_argument("--topic", required=True)
    p.add_argument("--start-index", type=int)
    p.add_argument("--end-index", type=int)
    p.add_argument("--allow-undecodable", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--index-dir", **index_dir)
    p.set_defaults(fn=cmd_mp4)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
