import hashlib
import http.server
import os
import threading

import pytest

import mcap_frames as mf
from tests import fixtures as fx

N = 120


@pytest.fixture(scope="session")
def clean_units():
    return fx.encode_h264(N, gop=10)


@pytest.fixture(scope="session")
def reference(clean_units):
    return fx.sequential_pictures(clean_units)


def md5(rgb):
    return hashlib.md5(rgb.tobytes()).hexdigest()


@pytest.mark.parametrize("compression", ["none", "zstd", "lz4"])
@pytest.mark.parametrize("kind", ["protobuf", "ros2"])
def test_frames_match_sequential_decode(tmp_path, clean_units, reference, compression, kind):
    cams = [fx.Camera("/cam/a", clean_units, kind), fx.Camera("/cam/b", fx.encode_h264(40, gop=5), kind)]
    w = fx.write_recording(str(tmp_path / "r.mcap"), cams, compression=compression)
    with mf.open(w.path) as rec:
        assert rec.video_topics == ["/cam/a", "/cam/b"]
        v = rec.video("/cam/a")
        assert len(v) == N and v.codec == "h264"
        wanted = [0, 1, 9, 10, 11, 57, 99, 119]
        for f in v.frames(wanted):
            assert fx.read_number(f.image) == f.index
            assert md5(f.image) == reference[f.index]
            assert f.decoded_from == f.index - f.index % 10
            assert f.stamp_ns == fx.T0 + f.index * 1_000_000_000 // fx.FPS
        assert [f.index for f in v.frames([57, 3, 57])] == [57, 3, 57]       # request order, duplicates kept


def test_sparse_access_reads_little(tmp_path, clean_units):
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", clean_units)], compression="none")
    total = os.path.getsize(w.path)
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        len(v)                                               # build the index first (message index records)
        before = rec.stats.bytes
        v.frame(57)
        assert rec.stats.bytes - before < total / 5          # only frames near 57, not the whole stream


def test_index_at(tmp_path, clean_units):
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", clean_units)])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        t = int(v.log_times_ns[30])
        assert v.index_at(t) == 30
        assert v.index_at(t + 1, "before") == 30
        assert v.index_at(t + 1, "after") == 31
        assert v.index_at(t - 1) == 30


def test_missing_keyframe_is_reported_not_shifted(tmp_path, clean_units, reference):
    units = fx.drop(clean_units, [40])                       # an IDR: source frames 41..49 lose their reference
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    numbers = w.numbers["/cam"]
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        for m in range(40, 49):                              # messages carrying source frames 41..49
            with pytest.raises(mf.FrameNotDecodable):
                v.frame(m)
        for f in v.frames([39, 49, 50, 100, 118]):
            assert fx.read_number(f.image) == numbers[f.index]
            assert md5(f.image) == reference[numbers[f.index]]
        assert [n for n in range(len(v)) if v.decodable(n) is False] == list(range(40, 49))
        assert v.decodable(45) is False and v.decodable(60) is True
        # the same answers once the scan is available
        with pytest.raises(mf.FrameNotDecodable):
            v.frame(44)
        assert fx.read_number(v.frame(60).image) == numbers[60]
    rep = mf.check(w.path)
    issues = {i.code: i for i in rep.topics["/cam"].issues}
    assert not rep.ok and issues["missing-reference"].frames == [(40, 48)]


def test_recording_starting_mid_gop(tmp_path, clean_units):
    units = [u for u in clean_units if u.number >= 5]
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        with pytest.raises(mf.NoKeyframeBefore):
            v.frame(2)
        assert fx.read_number(v.frame(10).image) == 15
    rep = mf.check(w.path)
    codes = {i.code: i for i in rep.topics["/cam"].issues}
    assert rep.ok and codes["starts-mid-gop"].frames == [(0, 4)]


def test_parameter_sets_only_once(tmp_path, clean_units, reference):
    units = fx.strip_params(clean_units)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        for f in v.frames([5, 55, 119]):
            assert md5(f.image) == reference[f.index]
        run = v.encoded(55)
        assert run.first_index == 50 and run.units[0].startswith(b"\x00\x00\x00\x01")
    rep = mf.check(w.path)
    assert rep.ok and "params-not-repeated" in {i.code for i in rep.topics["/cam"].issues}


def test_b_frames_are_refused(tmp_path):
    units = fx.encode_h264(40, gop=10, bframes=2)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    with mf.open(w.path) as rec:
        with pytest.raises(mf.UnsupportedStream):
            rec.video("/cam").frame(13)
    rep = mf.check(w.path)
    assert not rep.ok and "b-frames" in {i.code for i in rep.topics["/cam"].issues}


def test_jpeg_stream(tmp_path):
    units = fx.encode_jpeg(20)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, kind="ros2", fmt="jpeg")])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        assert v.codec == "jpeg"
        assert [fx.read_number(f.image) for f in v.frames([3, 17])] == [3, 17]
        assert v.encoded(4).units == [units[4].data]


def test_frames_skipped_warning(tmp_path, clean_units):
    times = [fx.T0 + i * 33_333_333 + (33_333_333 if i >= 60 else 0) for i in range(N)]
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", clean_units, times_ns=times)])
    rep = mf.check(w.path)
    gap = {i.code: i for i in rep.topics["/cam"].issues}["frames-skipped"]
    assert gap.level == "warning" and gap.frames == [(59, 60)]


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        f = open(path, "rb")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            return f
        start, end = (int(x) for x in rng.split("=")[1].split("-"))
        f.seek(start)
        self._remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Length", str(self._remaining))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        outputfile.write(source.read() if n is None else source.read(n))


def test_http_source(tmp_path, clean_units, reference):
    fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", clean_units)], compression="none")
    handler = lambda *a, **k: _RangeHandler(*a, directory=str(tmp_path), **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/r.mcap"
        with mf.open(url) as rec:
            opened = rec.stats.reads
            f = rec.video("/cam").frame(57)
            assert md5(f.image) == reference[57]
            assert opened <= 2 and rec.stats.reads - opened <= 12
    finally:
        server.shutdown()


def test_requests_larger_than_the_payload_cache(tmp_path, monkeypatch, clean_units, reference):
    import mcap_frames.video as video
    monkeypatch.setattr(video, "_PAYLOAD_CACHE", 8)          # far smaller than one run from keyframe to target
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", fx.strip_params(clean_units))])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        for f in v.frames([3, 19, 58, 59, 119]):
            assert md5(f.image) == reference[f.index]
        assert len(v._payloads(range(0, 40))) == 40


needs_x265 = pytest.mark.skipif("libx265" not in __import__("av").codecs_available, reason="no libx265 encoder")


@needs_x265
def test_h265_frames_match_sequential_decode(tmp_path):
    units = fx.encode_h265(40, gop=10)
    reference = fx.sequential_pictures(units, "h265")
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, fmt="h265")])
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        assert v.codec == "h265"
        for f in v.frames([0, 9, 13, 39]):
            assert fx.read_number(f.image) == f.index and md5(f.image) == reference[f.index]
    rep = mf.check(w.path)
    assert rep.ok and "b-frames" not in {i.code for i in rep.topics["/cam"].issues}


@needs_x265
def test_h265_b_frames_are_refused(tmp_path):
    units = fx.encode_h265(40, gop=10, bframes=2)
    assert [u.number for u in units[:4]] != [0, 1, 2, 3]          # picture order differs from message order
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, fmt="h265")])
    with mf.open(w.path) as rec:                                  # without a scan: found on the way to frame n
        with pytest.raises(mf.UnsupportedStream, match="B-frames"):
            rec.video("/cam").frame(13)
    with mf.open(w.path) as rec:                                  # with a scan
        rec.scan()
        with pytest.raises(mf.UnsupportedStream, match="B-frames"):
            rec.video("/cam").frame(13)
    rep = mf.check(w.path)
    assert not rep.ok and "b-frames" in {i.code for i in rep.topics["/cam"].issues}
