import json

import av
import pytest

import mcap_frames as mf
from mcap_frames.cli import main
from mcap_frames.source import Source
from tests import fixtures as fx


class _Bytes(Source):
    def __init__(self, data):
        super().__init__()
        self.data, self.size, self.name = data, len(data), "bytes"

    def _read(self, offset, size):
        return self.data[offset:offset + size]


def test_read_many_merges_nearby_ranges():
    s = _Bytes(bytes(range(256)) * 100)
    out = s.read_many([(100, 4), (0, 2), (110, 3), (20000, 5)], merge_gap=16)
    assert out == [s.data[100:104], s.data[0:2], s.data[110:113], s.data[20000:20005]]
    assert s.stats.reads == 3                       # (0,2) and (100..113) are 98 bytes apart: two reads, plus one


@pytest.fixture(scope="module")
def damaged(tmp_path_factory):
    units = fx.drop(fx.encode_h264(90, gop=10), [40])
    return fx.write_recording(str(tmp_path_factory.mktemp("d") / "r.mcap"), [fx.Camera("/cam", units)])


def test_export_frames_lists_undecodable(tmp_path, damaged):
    with mf.open(damaged.path) as rec:
        rows = mf.export_frames(rec.video("/cam"), [10, 44, 60], tmp_path)
    assert [r["status"] for r in rows] == ["ok", "not_decodable", "ok"]
    manifest = json.loads((tmp_path / "frames.json").read_text())
    assert manifest["frames"][1]["file"] is None and (tmp_path / "00000060.jpg").exists()


def test_export_frames_survives_a_decoder_failure(tmp_path, damaged, monkeypatch):
    """A run the decoder rejects is reported for its frames; the other frames are still written."""
    import mcap_frames.decode as decode_mod
    real = decode_mod.decode_run

    def failing(units, codec, want, threads=1):
        if len(units) > 5:                                    # the run for frame 66 (keyframe 59 .. 66)
            raise mf.FrameNotDecodable("decoder produced 7 pictures for 8 access units")
        return real(units, codec, want, threads)

    monkeypatch.setattr(decode_mod, "decode_run", failing)
    with mf.open(damaged.path) as rec:
        rows = mf.export_frames(rec.video("/cam"), [10, 44, 66, 52, 10], tmp_path)   # 66 and 52: other runs
    assert [r["status"] for r in rows] == ["ok", "not_decodable", "not_decodable", "ok", "ok"]
    assert [r["file"] for r in rows] == ["00000010.jpg", None, None, "00000052.jpg", "00000010.jpg"]


def test_export_frames_reports_frames_before_the_first_keyframe(tmp_path):
    units = fx.encode_h264(30, gop=10)[3:]                     # starts mid-GOP
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    with mf.open(w.path) as rec:
        rows = mf.export_frames(rec.video("/cam"), [2, 8], tmp_path / "out")
    assert [r["status"] for r in rows] == ["no_keyframe_before", "ok"]


@pytest.mark.parametrize("scanned", [False, True])
def test_mp4_refuses_damaged_range_unless_allowed(tmp_path, damaged, scanned):
    with mf.open(damaged.path) as rec:
        v = rec.video("/cam")
        if scanned:
            rec.scan()
        with pytest.raises(mf.FrameNotDecodable):
            mf.to_mp4(v, str(tmp_path / "a.mp4"), 30, 60)
        info = mf.to_mp4(v, str(tmp_path / "b.mp4"), 52, 80)
    assert info["first_frame"] == 49 and info["frames"] == 32          # moved back to the keyframe (source 50)
    with av.open(str(tmp_path / "b.mp4")) as c:
        pics = [fx.read_number(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]
    assert pics == damaged.numbers["/cam"][49:81]


def test_mp4_reads_only_the_range(tmp_path):
    """Without a scan, a short clip reads the frames it needs, not the whole topic."""
    units = fx.encode_h264(600, gop=30)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)], compression="zstd",
                           chunk_size=32 * 1024)
    size = (tmp_path / "r.mcap").stat().st_size
    with mf.open(w.path) as rec:
        v = rec.video("/cam")
        len(v)                                                     # opened and indexed
        before = rec.stats.bytes
        info = mf.to_mp4(v, str(tmp_path / "clip.mp4"), 305, 340)
        assert v._scan is None                                     # no whole-topic scan
        assert rec.stats.bytes - before < size / 2                 # a scan would read every chunk
    assert info["first_frame"] == 300 and info["frames"] == 41
    with av.open(str(tmp_path / "clip.mp4")) as c:
        assert [fx.read_number(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)] == list(range(300, 341))


def test_mp4_refuses_b_frames_without_a_scan(tmp_path):
    units = fx.encode_h264(40, gop=10, bframes=2)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)])
    with mf.open(w.path) as rec, pytest.raises(mf.UnsupportedStream, match="B-frames"):
        mf.to_mp4(rec.video("/cam"), str(tmp_path / "b.mp4"), 12, 25)


def test_mp4_rejects_an_empty_range(tmp_path, damaged):
    with mf.open(damaged.path) as rec, pytest.raises(ValueError):
        mf.to_mp4(rec.video("/cam"), str(tmp_path / "e.mp4"), 70, 60)


def test_cli_check_exit_code(tmp_path, damaged, capsys):
    clean = fx.write_recording(str(tmp_path / "c.mcap"), [fx.Camera("/cam", fx.encode_h264(30))])
    assert main(["check", clean.path]) == 0
    assert main(["check", damaged.path, "--json", str(tmp_path / "r.json")]) == 1
    assert "missing-reference" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())[0]["ok"] is False
