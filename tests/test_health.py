"""Health check: each kind of timing or loss problem, reproduced on its own.

Timelines are built on a 30 fps grid with a little noise. A frame "skipped by the camera" leaves a slot of the
grid empty (the encoder never saw it, frame_num stays continuous); a frame "lost after encoding" is a message
removed from the stream (frame_num shows the gap); "jitter" moves a single frame's timestamp off its slot.
"""

import numpy as np
import pytest

import mcap_frames as mf
from mcap_frames.health import _explain_reference_gap
from tests import fixtures as fx

N = 200
P = 33_333_333


@pytest.fixture(scope="module")
def units():
    return fx.encode_h264(N, gop=10)


def grid(slots, noise_ms=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return [fx.T0 + s * P + int(rng.uniform(-noise_ms, noise_ms) * 1e6) for s in slots]


def check(tmp_path, units, stamps=None, times=None, kind="protobuf"):
    w = fx.write_recording(str(tmp_path / "r.mcap"),
                           [fx.Camera("/cam", units, kind=kind, times_ns=times, stamps_ns=stamps)])
    rep = mf.check(w.path)
    issues = rep.topics["/cam"].issues
    assert all(i.evidence for i in issues)                       # every issue says what it is based on
    return {i.code: i for i in issues}


def test_clean_timeline_has_no_issue(tmp_path, units):
    assert check(tmp_path, units, stamps=grid(range(N))) == {}


def test_late_and_early_frames_are_jitter(tmp_path, units):
    stamps = grid(range(N))
    stamps[30] = stamps[31]                     # stamped with the next frame's time (same stamp twice)
    stamps[80] += int(0.7 * P)                  # late, alone
    stamps[120] -= int(0.7 * P)                 # early: the long interval comes after it
    stamps[150] += int(0.9 * P)                 # two late frames close together
    stamps[152] += int(0.9 * P)
    issues = check(tmp_path, units, stamps=stamps)
    assert "frames-skipped" not in issues
    assert issues["timestamp-jitter"].level == "info"
    assert {a for a, _b in issues["timestamp-jitter"].frames} == {29, 79, 120, 149, 151}
    assert issues["timestamp-not-increasing"].frames == [(30, 31)]


def test_frames_skipped_by_the_camera(tmp_path, units):
    slots = list(range(N + 3))
    del slots[150:152]                           # two slots empty in a row
    del slots[60]
    stamps = grid(slots[:N])
    stamps[100] += int(0.8 * P)                  # and some jitter, which must not count
    issues = check(tmp_path, units, stamps=stamps)
    skipped = issues["frames-skipped"]
    assert skipped.level == "warning" and skipped.frames == [(59, 60), (148, 149)]
    assert "about 3 frames" in skipped.message and "capture timeline" in skipped.message
    assert skipped.inferred and skipped.evidence.startswith("capture timestamps, 98% of intervals")
    assert "missing-reference" not in issues                     # frame_num is continuous: nothing to decode wrong


def test_skips_close_together_are_counted_once(tmp_path, units):
    slots = [s for s in range(N + 2) if s not in (70, 73)]
    issues = check(tmp_path, units, stamps=grid(slots[:N]))
    assert "about 2 frames" in issues["frames-skipped"].message


def test_lost_reference_frame(tmp_path, units):
    kept = fx.drop(units, [45])                  # a P-frame lost after encoding
    issues = check(tmp_path, kept, stamps=grid([u.number for u in kept]))
    ref = issues["missing-reference"]
    assert ref.frames == [(45, 48)]
    assert "before frame 45 1 reference frame went missing (frame_num 6, expected 5)" in ref.message
    assert "1 of them where frames were lost after encoding" in issues["frames-skipped"].message
    assert "scene-cut" in issues["irregular-gop"].message           # the shorter GOP is not blamed on a keyframe


def test_lost_keyframe(tmp_path, units):
    kept = fx.drop(units, [40, 130])             # two keyframes lost after encoding
    issues = check(tmp_path, kept, stamps=grid([u.number for u in kept]))
    ref = issues["missing-reference"]
    assert ref.frames == [(40, 48), (129, 137)]
    assert not ref.inferred and "frame_num" in ref.evidence
    assert "before frame 40 a keyframe went missing (frame_num restarts at 1 without a keyframe)" in ref.message
    assert "2 keyframes were lost" in issues["irregular-gop"].message
    assert issues["frames-skipped"].frames == [(39, 40), (128, 129)]


def test_lost_keyframe_and_the_frames_after_it(tmp_path, units):
    kept = fx.drop(units, [70, 71, 72])
    ref = check(tmp_path, kept, stamps=grid([u.number for u in kept]))["missing-reference"]
    assert "a keyframe and 2 reference frames went missing (frame_num restarts at 3" in ref.message


def test_capture_timestamps_going_backwards(tmp_path, units):
    stamps = grid(range(N))
    stamps[90], stamps[91] = stamps[91], stamps[90]
    issue = check(tmp_path, units, stamps=stamps)["timestamp-not-increasing"]
    assert issue.frames == [(90, 91)] and "0 frames carry the same" in issue.message and "1 an earlier" in issue.message


def test_log_time_is_used_without_capture_timestamps(tmp_path, units):
    slots = [s for s in range(N + 1) if s != 100]
    issues = check(tmp_path, units, times=grid(slots[:N], noise_ms=0), kind="pb_image")
    assert issues["frames-skipped"].frames == [(99, 100)] and "log_time timeline" in issues["frames-skipped"].message
    assert issues["frames-skipped"].evidence.startswith("log_time, ")


def test_long_pause(tmp_path, units):
    slots = list(range(100)) + list(range(400, 500))        # ten seconds without frames
    skipped = check(tmp_path, units, stamps=grid(slots))["frames-skipped"]
    assert "about 300 frames" in skipped.message


@pytest.mark.parametrize("expected, found, max_fn, text", [
    (10, 1, 16, "a keyframe went missing"),                  # restart: the keyframe itself
    (10, 3, 16, "a keyframe and 2 reference frames"),
    (5, 7, 16, "2 reference frames went missing"),           # plain jump forward
    (15, 1, 16, "a keyframe went missing"),                   # after a wrap both readings fit: the smaller one
    (250, 2, 256, "a keyframe and 1 reference frame went"),
])
def test_reference_gap_explanation(expected, found, max_fn, text):
    assert text in _explain_reference_gap(7, expected, found, max_fn)[0]


def test_bursts_are_not_analysed(tmp_path, units):
    # frames delivered four at a time: the timestamps say nothing about skipped frames
    stamps = [fx.T0 + int((k // 4 * 4 + (0, 1, 1.5, 1.5)[k % 4]) * P) for k in range(N)]
    issues = check(tmp_path, units, stamps=stamps)
    assert "timing-irregular" in issues and "frames-skipped" not in issues


def test_slow_camera_on_a_coarse_clock(tmp_path, units):
    # 29.8 fps, stamped by a 60 Hz clock: every few frames an interval of 1.5 periods, and no frame is missing
    tick = P / 2
    stamps = [fx.T0 + int(round(k * P * 1.0067 / tick) * tick) for k in range(N)]
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units, stamps_ns=stamps)])
    rep = mf.check(w.path).topics["/cam"]
    assert "frames-skipped" not in {i.code for i in rep.issues}
    assert rep.fps == pytest.approx(29.8, abs=0.05)


def test_summary_and_json_show_the_evidence(tmp_path, units):
    kept = fx.drop(units, [40])
    w = fx.write_recording(str(tmp_path / "r.mcap"),
                           [fx.Camera("/cam", kept, stamps_ns=grid([u.number for u in kept]))])
    rep = mf.check(w.path)
    text = rep.summary()
    assert "missing-reference (from NAL unit and slice headers (frame_num, parameter sets)):" in text
    assert "frames-skipped (inferred from capture timestamps, " in text
    topic = rep.to_dict()["topics"]["/cam"]
    assert topic["timing"]["clock"] == "capture" and topic["timing"]["regular_share"] > 0.9
    assert {i["code"]: i["inferred"] for i in topic["issues"]}["frames-skipped"] is True
