import io
import threading
import zipfile
from types import SimpleNamespace

import pytest
import requests

from src.download import select
from src.remote_zip import HttpRange, RangeFile, list_members, read_member


def archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for i, (name, data) in enumerate(members.items()):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED
            info.extra = b"\x99\x99\x04\x00abcd"  # unknown extra field, as some zip tools write
            z.writestr(info, data)
    return buffer.getvalue()


class Server:
    """Serves byte ranges of an in-memory file and counts the requests."""

    def __init__(self, data: bytes):
        self.data, self.requests = data, 0

    def fetch(self, start: int, end: int) -> bytes:
        self.requests += 1
        return self.data[start : end + 1]


MEMBERS = {"MOT17/train/MOT17-02-FRCNN/seqinfo.ini": b"[Sequence]\nname=MOT17-02-FRCNN\n",
           "MOT17/train/MOT17-02-FRCNN/img1/000001.jpg": bytes(range(256)) * 50,
           "MOT17/train/MOT17-02-FRCNN/gt/gt.txt": b"1,1,10,20,30,40,1,1,1.0\n" * 100,
           "MOT17/train/MOT17-02-DPM/img1/000001.jpg": b"same frame, other copy",
           "MOT17/test/MOT17-01-FRCNN/img1/000001.jpg": b"no labels"}


def test_members_are_read_with_one_request_each():
    server = Server(archive(MEMBERS))
    members = list_members(server.fetch, len(server.data))
    assert set(members) == set(MEMBERS)
    for name, content in MEMBERS.items():
        before = server.requests
        assert read_member(server.fetch, members[name]) == content
        assert server.requests == before + 1


def test_corrupted_member_is_detected():
    data = bytearray(archive(MEMBERS))
    server = Server(bytes(data))
    info = list_members(server.fetch, len(data))["MOT17/train/MOT17-02-FRCNN/img1/000001.jpg"]
    data[info.header_offset + 100] ^= 0xFF  # flip one byte inside the stored frame
    with pytest.raises(ValueError, match="CRC"):
        read_member(Server(bytes(data)).fetch, info)


def test_range_file_reads_across_blocks():
    data = bytes(range(256)) * 10
    f = RangeFile(Server(data).fetch, len(data), block=100)
    f.seek(250)
    assert f.read(120) == data[250:370]
    f.seek(-10, io.SEEK_END)
    assert f.read() == data[-10:]
    assert f.read(5) == b""


def test_select_keeps_one_variant_of_the_training_sequences():
    wanted = select(MEMBERS, ["MOT17-02"], "FRCNN")
    assert sorted(str(p) for p in wanted.values()) == ["MOT17-02/gt/gt.txt", "MOT17-02/img1/000001.jpg",
                                                         "MOT17-02/seqinfo.ini"]
    with pytest.raises(ValueError, match="MOT17-04"):
        select(MEMBERS, ["MOT17-02", "MOT17-04"], "FRCNN")


class FlakySession:
    """Drops the connection on the first two requests, then answers with a partial response."""

    calls = 0

    def get(self, url, headers, timeout):
        FlakySession.calls += 1
        if FlakySession.calls <= 2:
            raise requests.ConnectionError("Remote end closed connection without response")
        return SimpleNamespace(status_code=206, content=b"data", raise_for_status=lambda: None)


def test_dropped_connections_are_retried(monkeypatch):
    monkeypatch.setattr("src.remote_zip.time.sleep", lambda seconds: None)
    monkeypatch.setattr("src.remote_zip.requests.Session", FlakySession)
    remote = object.__new__(HttpRange)  # skip the HEAD request of __init__
    remote.url, remote.timeout, remote.retries, remote._local = "http://x", 1, 5, threading.local()
    assert remote.fetch(0, 3) == b"data"
    assert FlakySession.calls == 3
