"""HTTP sources: connection reuse, retries, timeouts, redirects, and servers that refuse HEAD."""

import http.server
import os
import threading
import time

import pytest

import mcap_frames as mf
from mcap_frames.source import HttpSource
from tests import fixtures as fx


class Server:
    """A range-capable HTTP/1.1 server with knobs for misbehaving."""

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self.connections = 0
        self.requests = 0
        self.fail_next = 0            # answer this many GETs with 503
        self.drop_next = 0            # close the connection without answering this many GETs
        self.delay = 0.0              # seconds before answering a GET
        self.allow_head = True
        self.ignore_range = False
        self.content_range = True     # send Content-Range with 206 (some minimal servers do not)
        self.redirect_to = "/file.mcap"
        self.authorization = []       # Authorization header of every GET
        self.lock = threading.Lock()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"             # keep connections open

            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                with server.lock:
                    server.connections += 1

            def do_HEAD(self):
                if not server.allow_head:
                    self.send_response(403)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(server.size))
                self.end_headers()

            def do_GET(self):
                with server.lock:
                    server.requests += 1
                    server.authorization.append(self.headers.get("Authorization"))
                    drop = server.drop_next > 0
                    fail = not drop and server.fail_next > 0
                    server.drop_next -= drop
                    server.fail_next -= fail
                if self.path.startswith(("/redirect", "/loop")):
                    self.send_response(307)
                    self.send_header("Location", "/loop" if self.path.startswith("/loop") else server.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if not self.path.startswith("/file.mcap"):
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if drop:
                    self.close_connection = True
                    self.connection.close()
                    return
                if fail:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                time.sleep(server.delay)
                a, b = (int(x) for x in self.headers["Range"].split("=")[1].split("-"))
                with open(server.path, "rb") as f:
                    if server.ignore_range:
                        data = f.read()
                        self.send_response(200)
                    else:
                        f.seek(a)
                        data = f.read(b - a + 1)
                        self.send_response(206)
                        if server.content_range:
                            self.send_header("Content-Range", f"bytes {a}-{a + len(data) - 1}/{server.size}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.handle_error = lambda request, address: None    # clients that time out and hang up
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/file.mcap"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def recording(tmp_path):
    units = fx.encode_h264(60, gop=10)
    w = fx.write_recording(str(tmp_path / "r.mcap"), [fx.Camera("/cam", units)], compression="none")
    return w.path, fx.sequential_pictures(units)


@pytest.fixture
def server(recording):
    s = Server(recording[0])
    yield s
    s.close()


def frames_ok(url, reference, **kwargs):
    import hashlib
    with mf.open(HttpSource(url, **kwargs)) as rec:
        v = rec.video("/cam")
        for f in v.frames([5, 33, 59]):
            assert hashlib.md5(f.image.tobytes()).hexdigest() == reference[f.index]
        return rec.stats


def test_connections_are_reused(server, recording):
    stats = frames_ok(server.url, recording[1])
    assert stats.reads >= 5
    assert server.connections <= HttpSource.concurrency + 1        # one per thread, not one per request


def test_size_without_head(server, recording):
    server.allow_head = False                                     # presigned GET URLs reject HEAD
    frames_ok(server.url, recording[1])


def test_transient_errors_are_retried(server, recording):
    server.fail_next = 2
    server.drop_next = 1
    frames_ok(server.url, recording[1], backoff=0.01)
    assert server.requests > 5


def test_persistent_errors_are_reported(server):
    src = HttpSource(server.url, retries=2, backoff=0.01)
    server.fail_next = 10
    with pytest.raises(IOError, match="503"):
        src.read(0, 16)


def test_read_timeout(server):
    src = HttpSource(server.url, read_timeout=0.2, retries=1, backoff=0.01)
    server.delay = 1.0
    t = time.perf_counter()
    with pytest.raises(IOError, match="timed out"):
        src.read(0, 16)
    assert time.perf_counter() - t < 1.5


def test_redirect_is_followed(server, recording):
    frames_ok(server.url.replace("/file.mcap", "/redirect"), recording[1])


def test_server_without_range_support(server):
    server.ignore_range = True
    with pytest.raises(IOError, match="range"):
        mf.open(server.url)


def test_missing_file(server):
    with pytest.raises(IOError, match="404"):
        HttpSource(server.url.replace("/file.mcap", "/nope.mcap") + "?missing")


def test_server_without_content_range(server, recording):
    server.content_range = False                   # the size then comes from HEAD
    frames_ok(server.url, recording[1])


def test_credentials_stay_on_the_original_host(server, recording):
    other = Server(recording[0])
    try:
        server.redirect_to = other.url                             # an absolute URL on another host (port)
        frames_ok(server.url.replace("/file.mcap", "/redirect"), recording[1],
                  headers={"Authorization": "Bearer secret"})
        assert server.authorization and set(server.authorization) == {"Bearer secret"}
        assert other.authorization and set(other.authorization) == {None}
    finally:
        other.close()


def test_redirect_loop_is_not_retried(server):
    with pytest.raises(IOError, match="redirects"):
        HttpSource(server.url.replace("/file.mcap", "/loop"), max_redirects=3, backoff=0.01)
    assert server.requests == 4                                    # one attempt: 1 + 3 redirects


def test_close_closes_every_connection(server, recording):
    src = HttpSource(server.url)
    with mf.open(src) as rec:
        rec.video("/cam").frames([5, 33, 59])
        conns = list(src._conns)
        assert conns
    assert not src._conns and all(c.sock is None for c in conns)
