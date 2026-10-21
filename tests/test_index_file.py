"""Index files: what a loaded index gives equals what reading the recording gives; stale ones are never used."""

import hashlib
import io
import json
import os

import numpy as np
import pytest

import mcap_frames as mf
from mcap_frames.cli import main as cli
from tests import fixtures as fx
from tests.test_http import Server


@pytest.fixture(scope="module")
def units():
    return fx.encode_h264(120, gop=10)


def md5(rgb):
    return hashlib.md5(rgb.tobytes()).hexdigest()


def cases(tmp, units):
    """Recordings of every kind the library reads, each with something to remember."""
    out = {}

    def add(name, cams, cut=None, **kw):
        path = os.path.join(tmp, f"{name}.mcap")
        fx.write_recording(path, cams, **kw)
        if cut:
            data = open(path, "rb").read()
            open(path, "wb").write(data[:int(len(data) * cut)])
        out[name] = path

    add("two-cameras", [fx.Camera("/a", units), fx.Camera("/b", fx.encode_h264(40, gop=5))])
    add("ros2-lz4", [fx.Camera("/a", units, kind="ros2")], compression="lz4")
    add("uncompressed", [fx.Camera("/a", units)], compression="none")
    add("keyframe-lost", [fx.Camera("/a", fx.drop(units, [40]))])
    add("params-once", [fx.Camera("/a", fx.strip_params(units))], compression="none")
    add("big-sei", [fx.Camera("/a", fx.with_big_sei(units))], compression="none")
    add("mid-gop", [fx.Camera("/a", units[5:])])
    add("no-message-index", [fx.Camera("/a", units)], message_index=False)
    add("unchunked", [fx.Camera("/a", units)], chunked=False, compression="none")
    add("truncated", [fx.Camera("/a", units)], cut=0.6, chunk_size=16 * 1024)
    add("jpeg", [fx.Camera("/a", fx.encode_jpeg(20), kind="ros2", fmt="jpeg")])
    add("b-frames", [fx.Camera("/a", fx.encode_h264(40, gop=10, bframes=2))])
    stamps = [fx.T0 + k * 33_333_333 + (33_333_333 if k >= 60 else 0) for k in range(len(units))]
    add("skipped", [fx.Camera("/a", units, stamps_ns=stamps)])
    return out


def scan_fields(v):
    s = v._scanned()
    return (s.keyframe.tolist(), s.carries_params.tolist(), s.decodable.tolist(), s.stamp_ns.tolist(), s.b_frames,
            s.gaps_allowed, s.params_first_seen, [tuple(g) for g in s.reference_gaps], s.params.tolist(), s.param_sets)


def topics_of(report):
    return report.to_dict()["topics"]


def test_loaded_index_gives_what_the_recording_gives(tmp_path, units):
    for name, path in cases(str(tmp_path), units).items():
        index = path + ".frames"
        with mf.open(path) as rec:
            rec.save_index(index)
        with mf.open(path) as direct, mf.open(path, index=index) as loaded:
            assert loaded.index_status == "loaded", name
            assert loaded.stats.reads == 1, name                         # only the end of the recording
            assert loaded.video_topics == direct.video_topics, name
            for t in direct.video_topics:
                a, b = direct.video(t), loaded.video(t)
                assert len(a) == len(b) and (a.log_times_ns == b.log_times_ns).all(), name
                assert [p.data for p in a._payloads(range(len(a)))] == [p.data for p in b._payloads(range(len(b)))]
                assert a.codec == b.codec and scan_fields(a) == scan_fields(b), name
            reads = loaded.stats.reads
            assert topics_of(mf.check(loaded)) == topics_of(mf.check(direct)), name
            assert loaded.stats.reads == reads, name                     # the check read nothing more
            for t in direct.video_topics:
                a, b = direct.video(t), loaded.video(t)
                if a.codec == "h264" and a._scanned().b_frames:
                    continue
                wanted = list(range(0, len(a), 7)) + [len(a) - 1]
                got_a = {f.index: md5(f.image) for f in a.iter_frames(wanted, on_error="skip")}
                got_b = {f.index: md5(f.image) for f in b.iter_frames(wanted, on_error="skip")}
                assert got_a == got_b and got_a, name


def test_parameter_sets_come_from_the_index(tmp_path, units):
    path = cases(str(tmp_path), units)["params-once"]
    with mf.open(path) as rec:
        rec.save_index(path + ".frames")
    with mf.open(path, index=path + ".frames") as rec:
        v = rec.video("/a")
        f = v.frame(95)
        assert fx.read_number(f.image) == 95
        assert rec.stats.reads == 2                     # the tail, then frames 90..95 in one read; not frame 0


def test_locating_only_index(tmp_path, units):
    path = cases(str(tmp_path), units)["keyframe-lost"]
    with mf.open(path) as rec:
        rec.save_index(path + ".locate", scan=False)
        rec.save_index(path + ".full")
    with mf.open(path, index=path + ".locate") as rec:
        assert rec.index_status == "loaded"
        v = rec.video("/a")
        assert v._scan is None
        with pytest.raises(mf.FrameNotDecodable):
            v.frame(44)                                 # found without a scan, on the way from the keyframe
        assert fx.read_number(v.frame(60).image) == 61
        rec.scan()
        rec.save_index(path + ".later")
    full, later = np.load(path + ".full"), np.load(path + ".later")
    assert sorted(full.files) == sorted(later.files)
    for k in full.files:
        assert np.array_equal(full[k], later[k]), k


def rewrite(index_path, change):
    a = dict(np.load(index_path, allow_pickle=False))
    change(a)
    with open(index_path, "wb") as f:
        np.savez_compressed(f, **a)


@pytest.mark.parametrize("damage, reason", [
    ("other-recording", "bytes, the index was made for"),
    ("same-size-other-end", "end of the recording differs"),
    ("truncated-index", "unreadable index file"),
    ("other-version", "version 99"),
    ("short-array", "differ in length"),
    ("chunk-out-of-range", "not in the index"),
    ("frame-outside-chunk", "outside its chunk"),
    ("sha256", "sha256 differs"),
])
def test_stale_index_is_not_used(tmp_path, units, damage, reason):
    path = str(tmp_path / "r.mcap")
    index = path + ".frames"
    fx.write_recording(path, [fx.Camera("/a", units)], compression="none")
    with mf.open(path) as rec:
        rec.save_index(index, sha256="aa" if damage == "sha256" else None)
    if damage == "other-recording":
        fx.write_recording(path, [fx.Camera("/a", units[:60])], compression="none")
    elif damage == "same-size-other-end":                   # one byte of a picture near the end
        data = bytearray(open(path, "rb").read())
        at = bytes(data).index(units[100].data) + len(units[100].data) // 2
        assert at > len(data) - 64 * 1024
        data[at] ^= 0xFF
        open(path, "wb").write(bytes(data))
    elif damage == "truncated-index":
        data = open(index, "rb").read()
        open(index, "wb").write(data[: len(data) // 2])
    elif damage == "other-version":
        rewrite(index, lambda a: a.update(meta=np.array(json.dumps({**json.loads(str(a["meta"])), "version": 99}))))
    elif damage == "short-array":
        rewrite(index, lambda a: a.update({"t0/offset": a["t0/offset"][:-1]}))
    elif damage == "chunk-out-of-range":
        rewrite(index, lambda a: a["t0/chunk"].__setitem__(0, 999))
    elif damage == "frame-outside-chunk":
        rewrite(index, lambda a: a["t0/bound"].__setitem__(3, 2 ** 40))
    with mf.open(path, index=index, sha256="bb" if damage == "sha256" else None) as rec:
        assert rec.index_status.startswith("stale: ") and reason in rec.index_status, rec.index_status
        v = rec.video("/a")                              # read from the recording instead
        assert fx.read_number(v.frame(len(v) - 1).image) == len(v) - 1


def test_missing_index(tmp_path, units):
    path = str(tmp_path / "r.mcap")
    fx.write_recording(path, [fx.Camera("/a", units)])
    with mf.open(path, index=str(tmp_path / "nope.frames")) as rec:
        assert rec.index_status == "missing" and len(rec.video("/a")) == len(units)
    with mf.open(path) as rec:
        assert rec.index_status == "none"


def test_over_http(tmp_path, units):
    path = str(tmp_path / "r.mcap")
    fx.write_recording(path, [fx.Camera("/a", units)], compression="none")
    with mf.open(path) as rec:
        rec.save_index(path + ".frames")
    recording, index_server = Server(path), Server(path + ".frames")       # the index served from another host
    try:
        with mf.open(recording.url, index=index_server.url) as rec:
            assert rec.index_status == "loaded"
            assert recording.requests == 2               # the size probe and the tail
            mf.check(rec)
            assert recording.requests == 2
            assert fx.read_number(rec.video("/a").frame(57).image) == 57
    finally:
        recording.close()
        index_server.close()


def test_cli(tmp_path, units, capsys):
    data = tmp_path / "data"
    for sub, cut in (("a", None), ("b", [40])):                       # the same file name in two directories
        (data / sub).mkdir(parents=True)
        fx.write_recording(str(data / sub / "r.mcap"), [fx.Camera("/a", fx.drop(units, cut) if cut else units)])
    out = tmp_path / "idx"
    assert cli(["index", str(data), "--out-dir", str(out), "--check", str(tmp_path / "at-index.json")]) == 1
    assert (out / "a" / "r.mcap.frames").exists() and (out / "b" / "r.mcap.frames").exists()
    assert cli(["check", str(data), "--json", str(tmp_path / "direct.json")]) == 1
    assert cli(["check", str(data), "--index-dir", str(out), "--json", str(tmp_path / "indexed.json")]) == 1
    reports = [[r["topics"] for r in json.load(open(tmp_path / f"{n}.json"))]
               for n in ("direct", "indexed", "at-index")]
    assert reports[0] == reports[1] == reports[2]
    assert "index stale" not in capsys.readouterr().err
    assert cli(["info", str(data / "a" / "r.mcap"), "--index-dir", str(out / "a")]) == 0
    assert cli(["index", str(data / "a"), "--out-dir", str(tmp_path / "idx2"), "--no-scan"]) == 0
    meta = json.loads(str(np.load(tmp_path / "idx2" / "r.mcap.frames")["meta"]))
    assert [t["scanned"] for t in meta["topics"]] == [False]


def test_index_file_is_plain_npz(tmp_path, units):
    """Readable without this library: a zip of .npy arrays, no pickled objects."""
    path = str(tmp_path / "r.mcap")
    fx.write_recording(path, [fx.Camera("/a", units)])
    with mf.open(path) as rec:
        rec.save_index(path + ".frames")
    raw = open(path + ".frames", "rb").read()
    a = np.load(io.BytesIO(raw), allow_pickle=False)
    meta = json.loads(str(a["meta"]))
    assert meta["format"] == "mcap-frames-index" and meta["topics"][0]["frames"] == len(units)
    assert a["t0/flags"].dtype == np.uint8 and len(a["t0/flags"]) == len(units)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]          # written atomically


def test_topics_missing_from_the_index_are_read_from_the_recording(tmp_path, units):
    """An index that lacks a topic (written by an older version, say): that topic is read from the recording,
    and scanning it together with indexed topics reads each through its own container."""
    path = str(tmp_path / "r.mcap")
    b_units = [fx.Unit(u.number + 1000, u.data, u.key) for u in fx.drop(units, [40])]   # its own frame numbers
    fx.write_recording(path, [fx.Camera("/a", units), fx.Camera("/b", b_units)], outside_chunks=range(1050, 1055))
    index = path + ".frames"
    with mf.open(path) as rec:
        rec.save_index(index, scan=False)       # /b's messages outside chunks give it a chunk table of its own

    def keep_only_a(a):
        meta = json.loads(str(a["meta"]))
        meta["topics"] = [t for t in meta["topics"] if t["topic"] == "/a"]
        a["meta"] = np.array(json.dumps(meta))

    rewrite(index, keep_only_a)
    with mf.open(path) as direct, mf.open(path, index=index) as rec:
        assert rec.index_status == "loaded" and rec.video_topics == ["/a"]
        rec.scan(["/a", "/b"])
        for t in ("/a", "/b"):
            assert scan_fields(rec.video(t)) == scan_fields(direct.video(t)), t
        assert topics_of(mf.check(rec, ["/a", "/b"])) == topics_of(mf.check(direct, ["/a", "/b"]))
