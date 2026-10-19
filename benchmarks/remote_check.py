"""Check mcap-frames against a real object store (or any HTTP server with range support).

    python benchmarks/remote_check.py [--local copy.mcap] [--topic /camera/front] [--out result.json]

The URL (usually a presigned GET URL) is read from the MF_URL environment variable, or asked for without
echoing it. It is never printed or saved: presigned URLs carry a signature. The output only shows the host.

--local: a local copy of the same file. Health reports and decoded frames are then compared with it.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import statistics
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np

import mcap_frames as mf
from mcap_frames.source import HttpSource


class CountingHttpSource(HttpSource):
    """Counts new connections (the library itself does not expose this)."""

    def __init__(self, *args, **kwargs):
        self.new_connections = 0
        super().__init__(*args, **kwargs)

    def _connection(self, scheme, host):
        if (scheme, host) not in self._local.__dict__.get("conns", {}):
            self.new_connections += 1
        return super()._connection(scheme, host)


def md5(image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def delta(src, before):
    return {"requests": src.stats.reads - before[0], "MB": round((src.stats.bytes - before[1]) / 1e6, 2),
            "new_connections": src.new_connections - before[2]}


def snapshot(src):
    return src.stats.reads, src.stats.bytes, src.new_connections


URL = ""          # kept for hiding it in error messages


def main() -> int:
    global URL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", help="local copy of the same file, for comparison")
    ap.add_argument("--topic", help="video topic to use (default: the first one)")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", help="also write the results as JSON")
    a = ap.parse_args()

    url = URL = os.environ.get("MF_URL") or getpass.getpass("URL (input hidden): ").strip()
    parts = urlsplit(url)
    shown = f"{parts.scheme}://{parts.netloc}/…"
    res: dict = {"host": parts.netloc, "python": sys.version.split()[0], "mcap_frames": mf.__version__}
    print(f"checking {shown}")

    # 1-2. size, open, index
    src, t = timed(lambda: CountingHttpSource(url))
    res["probe"] = {"s": round(t, 3), "size_MB": round(src.size / 1e6, 1), "etag": bool(src.etag)}
    if a.local:
        res["probe"]["size_matches_local"] = src.size == os.path.getsize(a.local)
    print(f"  size: {src.size / 1e6:.1f} MB in {t:.2f} s")
    before = snapshot(src)
    rec, t = timed(lambda: mf.open(src))
    topic = a.topic or rec.video_topics[0]
    v = rec.video(topic)
    _, t2 = timed(lambda: len(v))
    res["open_and_index"] = {"s": round(t + t2, 3), "topics": len(rec.video_topics), "frames": len(v),
                             "codec": v.codec, **delta(src, before)}
    print(f"  open + index: {t + t2:.2f} s, {len(v)} frames ({v.codec}), {res['open_and_index']}")

    # 3. health check (one pass over all video topics)
    before = snapshot(src)
    rep, t = timed(lambda: mf.check(rec))
    res["check"] = {"s": round(t, 3), "ok": rep.ok, **delta(src, before)}
    if a.local:
        local_rep = mf.check(a.local)
        res["check"]["matches_local"] = (json.dumps(rep.to_dict()["topics"], sort_keys=True)
                                         == json.dumps(local_rep.to_dict()["topics"], sort_keys=True))
    print(f"  health check: {t:.2f} s, ok={rep.ok}, {res['check']}")

    # 4. frames
    try:
        usable = np.array([n for n in range(len(v)) if v.decodable(n)])
    except mf.McapFramesError:
        usable = np.arange(len(v))
    rng = np.random.default_rng(1)
    singles, single_t = sorted(rng.choice(usable, size=min(8, len(usable)), replace=False).tolist()), []
    rec2 = mf.open(CountingHttpSource(url))
    v2 = rec2.video(topic)
    len(v2)
    src2 = rec2.source
    before = snapshot(src2)
    for n in singles:
        _, t = timed(lambda: v2.frame(n))
        single_t.append(t)
    res["single_frame"] = {"median_ms": round(statistics.median(single_t) * 1000), "max_ms": round(max(single_t) * 1000),
                           "first_ms": round(single_t[0] * 1000), **delta(src2, before)}
    batch = sorted(rng.choice(usable, size=min(a.frames, len(usable)), replace=False).tolist())
    rec3 = mf.open(CountingHttpSource(url))
    v3 = rec3.video(topic)
    len(v3)
    before = snapshot(rec3.source)
    got, t = timed(lambda: v3.frames(batch))
    res["batch"] = {"frames": len(batch), "s": round(t, 3), **delta(rec3.source, before)}
    if a.local:
        with mf.open(a.local) as loc:
            ref = loc.video(topic).frames(batch)
        res["batch"]["pixels_match_local"] = [md5(x.image) for x in got] == [md5(y.image) for y in ref]
    print(f"  single frame: {res['single_frame']}")
    print(f"  {len(batch)} frames: {res['batch']}")

    # 5. throughput: 4 MB blocks, as many in flight as the source allows (how a file without indexes is walked)
    from concurrent.futures import ThreadPoolExecutor
    block = 4 << 20
    blocks = [(off, min(block, src.size - off)) for off in range(0, min(64 << 20, src.size), block)]
    before = snapshot(src)
    with ThreadPoolExecutor(src.concurrency) as pool:
        _, t = timed(lambda: list(pool.map(lambda r: src.read(*r), blocks)))
    res["throughput"] = {"s": round(t, 3), **delta(src, before)}
    res["throughput"]["MB_per_s"] = round(res["throughput"]["MB"] / t, 1)
    print(f"  throughput: {res['throughput']['MB_per_s']} MB/s ({res['throughput']})")

    # 6. error handling
    broken = urlunsplit(parts._replace(path=parts.path + ".missing"))     # signature no longer matches
    t0 = time.perf_counter()
    try:
        HttpSource(broken, retries=3, backoff=0.5)
        res["broken_url"] = {"error": None}
    except IOError as e:
        res["broken_url"] = {"error": str(e).replace(broken, "<url>")[-120:]}
    res["broken_url"]["s"] = round(time.perf_counter() - t0, 2)          # 4xx: no retries, so well under 1 s
    t0 = time.perf_counter()
    try:
        HttpSource(url, read_timeout=0.001, retries=3, backoff=0.1)
        res["tiny_timeout"] = {"error": None}
    except IOError as e:
        res["tiny_timeout"] = {"error": str(e).replace(url, "<url>")[-120:]}
    res["tiny_timeout"]["s"] = round(time.perf_counter() - t0, 2)
    print(f"  broken URL: {res['broken_url']}")
    print(f"  0.001 s read timeout: {res['tiny_timeout']}")

    text = json.dumps(res, ensure_ascii=False, indent=1)
    assert url not in text and (not parts.query or parts.query not in text)       # never leak the signature
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:                                  # error messages name the URL: hide the signature
        msg = f"{type(e).__name__}: {e}"
        print("failed:", msg.replace(URL, "<url>") if URL else msg, file=sys.stderr)
        raise SystemExit(2)
