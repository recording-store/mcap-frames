"""Situations from real-world recordings (met in practice, or found when checking how the code handles them),
reproduced with synthetic data."""

import hashlib

import pytest

import mcap_frames as mf
from tests import fixtures as fx


def md5(rgb):
    return hashlib.md5(rgb.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def units():
    return fx.encode_h264(90, gop=30)


@pytest.fixture(scope="module")
def reference(units):
    return fx.sequential_pictures(units)


@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_file_without_message_indexes(tmp_path, units, reference, compression):
    """Some writers leave out MessageIndex records: chunks are read once to locate the frames."""
    cams = [fx.Camera("/cam/a", units), fx.Camera("/cam/b", fx.encode_h264(30, gop=10))]
    w = fx.write_recording(str(tmp_path / "r.mcap"), cams, compression=compression, message_index=False)
    with mf.open(w.path) as rec:
        assert not rec.container.has_message_indexes
        assert rec.video_topics == ["/cam/a", "/cam/b"]
        assert [len(rec.video(t)) for t in rec.video_topics] == [90, 30]
        for f in rec.video("/cam/a").frames([0, 29, 31, 89]):
            assert md5(f.image) == reference[f.index]
    rep = mf.check(w.path)
    assert rep.ok and "no-message-index" in {i.code for i in rep.topics["/cam/a"].issues}


def test_h264_in_compressed_image_messages_without_timestamp(tmp_path, units, reference):
    """foxglove.CompressedImage used for H.264 video, with no timestamp in the message."""
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, kind="pb_image")])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        assert v.codec == "h264"
        f = v.frame(45)
        assert md5(f.image) == reference[45] and f.stamp_ns is None


def test_log_time_out_of_order_keeps_decode_order(tmp_path, units, reference):
    """Frame numbers follow the order messages were written (decode order), not log_time: a camera whose
    timestamps jitter backwards must still decode, and the jitter is reported."""
    times = [fx.T0 + i * 33_333_333 for i in range(len(units))]
    times[51] = times[50] - 1_000_000                       # written after frame 50, stamped 1 ms before it
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, times_ns=times)], chunk_size=1 << 20)
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        for f in v.frames([50, 51, 52, 59]):
            assert fx.read_number(f.image) == f.index and md5(f.image) == reference[f.index]
        assert v.log_times_ns[51] == times[51]
        assert v.index_at(times[51]) == 51 and v.index_at(times[50]) == 50
    rep = mf.check(w.path)
    assert "timestamp-backwards" in {i.code for i in rep.topics["/cam"].issues}


@pytest.mark.parametrize("keep", [0.35, 0.6, 0.97])
@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_truncated_recording(tmp_path, units, reference, keep, compression):
    """A recording cut off mid-write (no summary, last chunk incomplete): complete chunks are recovered."""
    w = fx.write_recording(str(tmp_path / "full.mcap"), [fx.Camera("/cam", units)], compression=compression,
                           chunk_size=16 * 1024)
    data = open(w.path, "rb").read()
    cut = tmp_path / "cut.mcap"
    cut.write_bytes(data[:int(len(data) * keep)])
    with mf.open(str(cut)) as rec:
        assert rec.container.recovered and rec.container.ignored_tail_bytes > 0
        v = rec.video("/cam")
        assert 0 < len(v) <= len(units)
        with mf.open(w.path) as full:
            assert (v.log_times_ns == full.video("/cam").log_times_ns[:len(v)]).all()
        for f in v.frames(sorted({0, len(v) // 2, len(v) - 1})):
            assert md5(f.image) == reference[f.index]
    rep = mf.check(str(cut))
    assert "truncated" in {i.code for i in rep.topics["/cam"].issues}


def test_not_an_mcap_file(tmp_path):
    p = tmp_path / "x.mcap"
    p.write_bytes(b"this is not a recording" * 10)
    with pytest.raises(mf.UnsupportedFile):
        mf.open(str(p))


@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_large_sei_before_picture_data(tmp_path, units, reference, compression):
    """Keyframes whose picture data starts beyond the bytes read when scanning are read again in full."""
    big = fx.with_big_sei(units)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", big)], compression=compression)
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        s = v.scan()
        assert list(s.keyframe.nonzero()[0]) == [0, 30, 60]
        assert (s.decodable == 1).all()
        for f in v.frames([0, 31, 89]):
            assert md5(f.image) == reference[f.index]


@pytest.mark.parametrize("cut", [None, 0.6])
def test_unchunked_recording(tmp_path, units, reference, cut):
    """Messages written straight into the data section, without chunks or indexes (ROS 2 'fastwrite'),
    complete or cut off while recording."""
    cams = [fx.Camera("/cam/a", units), fx.Camera("/cam/b", fx.encode_h264(30, gop=10))]
    w = fx.write_recording(str(tmp_path / "r.mcap"), cams, compression="none", chunked=False)
    path = w.path
    if cut:
        data = open(w.path, "rb").read()
        path = str(tmp_path / "cut.mcap")
        open(path, "wb").write(data[:int(len(data) * cut)])
    with mf.open(path) as rec:
        assert rec.video_topics == ["/cam/a", "/cam/b"]
        v = rec.video("/cam/a")
        n = len(v)
        assert n == 90 if cut is None else 0 < n < 90
        for f in v.frames(sorted({0, 31, n - 1})):
            assert md5(f.image) == reference[f.index]
    codes = {i.code for i in mf.check(path).topics["/cam/a"].issues}
    assert "unchunked" in codes and ("truncated" in codes) == bool(cut)
