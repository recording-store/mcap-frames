"""FFmpeg reads up to 64 bytes past the end of its input; packets must be padded (found as a rare crash on
real recordings). Every frame is placed right before an inaccessible page, so any over-read crashes."""

import os
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import ctypes, mmap, sys
    sys.path.insert(0, {root!r})
    from tests import fixtures as fx
    from mcap_frames.decode import decode_run

    libc = ctypes.CDLL(None)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    keep = []

    def guarded(data):
        page = mmap.PAGESIZE
        pages = (len(data) + page - 1) // page + 1
        m = mmap.mmap(-1, pages * page)
        base = ctypes.addressof(ctypes.c_char.from_buffer(m))
        assert libc.mprotect(base + (pages - 1) * page, page, 0) == 0
        start = (pages - 1) * page - len(data)
        m[start:start + len(data)] = data
        keep.append(m)
        return memoryview(m)[start:start + len(data)]

    units = [guarded(u.data) for u in fx.encode_h264(12, gop=10)]
    got = decode_run(units, "h264", [9])
    assert got[9].shape == (fx.H, fx.W, 3)
    print("ok")
""")


@pytest.mark.skipif(os.name != "posix", reason="uses mprotect")
def test_decoder_never_reads_past_the_input():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", SCRIPT.format(root=root)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, f"decoder crashed or failed (exit {r.returncode}): {r.stderr[-500:]}"
