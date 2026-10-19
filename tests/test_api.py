"""The public interface: streaming frames, skipping undecodable ones, lookups by capture time, what is exposed."""

import hashlib

import pytest

import mcap_frames as mf
import mcap_frames.decode as decode_mod
from tests import fixtures as fx

N = 120
P = 1_000_000_000 // fx.FPS


@pytest.fixture(scope="module")
def units():
    return fx.encode_h264(N, gop=10)


@pytest.fixture(scope="module")
def reference(units):
    return fx.sequential_pictures(units)


def md5(rgb):
    return hashlib.md5(rgb.tobytes()).hexdigest()


def test_iter_frames_decodes_as_it_goes(tmp_path, units, monkeypatch):
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    calls = []
    real = decode_mod.decode_run
    monkeypatch.setattr(decode_mod, "decode_run", lambda *a, **k: calls.append(1) or real(*a, **k))
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        it = v.iter_frames(range(0, N, 5), workers=1)            # 24 frames in 12 runs
        first = next(it)
        assert first.index == 0 and len(calls) == 1               # one run decoded, not twelve
        assert [f.index for f in it] == list(range(5, N, 5)) and len(calls) == 12
        it = v.iter_frames(range(0, N, 5), workers=4)
        next(it)
        assert len(calls) - 12 <= 5                               # a few runs ahead, not all of them


def test_iter_frames_matches_frames(tmp_path, units, reference):
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        wanted = [57, 3, 57, 119, 3, 60]                          # any order, duplicates kept
        streamed = list(v.iter_frames(wanted))
        assert [f.index for f in streamed] == wanted
        assert [md5(f.image) for f in streamed] == [md5(f.image) for f in v.frames(wanted)]
        assert all(md5(f.image) == reference[f.index] for f in streamed)


def test_undecodable_frames_raise_or_are_skipped(tmp_path, units, reference):
    kept = fx.drop(units, [40])                                    # messages 40..48 lose their keyframe
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", kept)])
    numbers = [u.number for u in kept]
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        wanted = [39, 41, 45, 49, 41, 100]
        it = v.iter_frames(wanted)
        with pytest.raises(mf.FrameNotDecodable):
            next(it)                                               # before anything is yielded
        got = v.frames(wanted, on_error="skip")
        assert [f.index for f in got] == [39, 49, 100]
        assert all(md5(f.image) == reference[numbers[f.index]] for f in got)
        with pytest.raises(IndexError):
            v.frames([3, N + 5], on_error="skip")                  # out of range is never skipped
        with pytest.raises(ValueError):
            v.frames([3], on_error="ignore")


def test_empty_images_are_skipped(tmp_path):
    units = fx.encode_jpeg(10)
    units[4] = fx.Unit(4, b"", True)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, kind="ros2", fmt="jpeg")])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        assert [f.index for f in v.frames([3, 4, 5], on_error="skip")] == [3, 5]
        with pytest.raises(mf.FrameNotDecodable):
            v.frames([3, 4, 5])


def test_lookup_by_capture_time(tmp_path, units):
    times = [fx.T0 + k * P for k in range(N)]
    stamps = [t - 40_000_000 for t in times]                       # taken 40 ms before they were logged
    stamps[31] = stamps[30]                                        # two frames with the same capture time
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, times_ns=times, stamps_ns=stamps)])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        assert list(v.stamps_ns) == stamps
        assert list(v.keyframes) == list(range(0, N, 10))
        assert v.index_at(stamps[50], clock="capture") == 50
        assert v.index_at(stamps[50]) == 49                        # log_time: 40 ms later than capture
        assert v.index_at(stamps[30], clock="capture") == 30       # equal times: the first of them ...
        assert v.index_at(stamps[30], clock="capture", mode="after") == 30
        assert v.index_at(stamps[30], clock="capture", mode="before") == 31   # ... or the last, for "before"
        assert v.index_at(stamps[30] + 1, clock="capture") == 30
        with pytest.raises(ValueError):
            v.index_at(stamps[5], clock="wall")


def test_capture_time_needs_timestamps(tmp_path, units):
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, kind="pb_image")])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        assert (v.stamps_ns == -1).all()
        with pytest.raises(mf.UnsupportedStream, match="capture timestamp"):
            v.index_at(fx.T0, clock="capture")


def test_public_surface(tmp_path, units):
    namespace = {}
    exec("from mcap_frames import *", namespace)
    assert "open" not in namespace and "check" in namespace       # the builtin open stays usable
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        for internal in ("index", "payloads", "scan", "container"):
            assert not hasattr(v, internal)
        assert not hasattr(rec, "container")
        for public in ("topic", "codec", "log_times_ns", "stamps_ns", "keyframes", "index_at", "decodable",
                       "frame", "frames", "iter_frames", "encoded"):
            assert hasattr(v, public)
