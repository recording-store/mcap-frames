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


def test_mp4_refuses_damaged_range_unless_allowed(tmp_path, damaged):
    with mf.open(damaged.path) as rec:
        v = rec.video("/cam")
        with pytest.raises(mf.FrameNotDecodable):
            mf.to_mp4(v, str(tmp_path / "a.mp4"), 30, 60)
        info = mf.to_mp4(v, str(tmp_path / "b.mp4"), 52, 80)
    assert info["first_frame"] == 49 and info["frames"] == 32          # moved back to the keyframe (source 50)
    with av.open(str(tmp_path / "b.mp4")) as c:
        pics = [fx.read_number(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]
    assert pics == damaged.numbers["/cam"][49:81]


def test_cli_check_exit_code(tmp_path, damaged, capsys):
    clean = fx.write_recording(str(tmp_path / "c.mcap"), [fx.Camera("/cam", fx.encode_h264(30))])
    assert main(["check", clean.path]) == 0
    assert main(["check", damaged.path, "--json", str(tmp_path / "r.json")]) == 1
    assert "missing-reference" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())[0]["ok"] is False
