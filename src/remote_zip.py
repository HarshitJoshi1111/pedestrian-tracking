"""Read single members of a large zip archive on a web server without downloading the whole archive.

A zip archive ends with a central directory that lists every member and the byte offset where it
starts. With HTTP range requests we read only that directory (a few MB at the end of the file) and
then only the byte ranges of the members we need. MOT17.zip is 5.5 GB because each sequence is stored
three times; the frames of the seven training sequences are about 1.2 GB.
"""
import io
import struct
import threading
import time
import zipfile
import zlib
from collections.abc import Callable

import requests

Fetch = Callable[[int, int], bytes]  # (first byte, last byte inclusive) -> bytes

_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # fixed 30-byte part of a member's local header
_LOCAL_SIGNATURE = b"PK\x03\x04"
_SLACK = 1024  # extra bytes requested so the variable-length local header fits in the same request


class HttpRange:
    """Byte ranges of one URL. Each thread gets its own HTTP session (sessions are not thread-safe)."""

    def __init__(self, url: str, timeout: float = 60, retries: int = 5):
        self.timeout, self.retries = timeout, retries
        self._local = threading.local()
        head = self._session().head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        self.url = head.url
        self.size = int(head.headers["Content-Length"])
        self.fetch(0, 3)  # fails early if the server ignores range requests

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def fetch(self, start: int, end: int) -> bytes:
        """One range request; dropped connections and server errors are retried with a new session."""
        for attempt in range(self.retries + 1):
            try:
                response = self._session().get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout)
                if response.status_code < 500:
                    break
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt == self.retries:
                    raise
                print(f"  retrying bytes {start}-{end} after: {error.__class__.__name__}", flush=True)
            del self._local.session  # the server may have closed a kept-alive connection
            time.sleep(min(2**attempt, 30))
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"{self.url} does not support range requests (status {response.status_code})")
        return response.content


class RangeFile(io.RawIOBase):
    """Seekable, read-only file on top of a Fetch function, so zipfile can read the central directory.

    Reads are served from cached 1 MB blocks: zipfile makes thousands of tiny reads while parsing the
    directory, which would otherwise be thousands of HTTP requests.
    """

    def __init__(self, fetch: Fetch, size: int, block: int = 1 << 20):
        super().__init__()
        self._fetch, self._size, self._block = fetch, size, block
        self._pos = 0
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = base + offset
        return self._pos

    def read(self, n: int = -1) -> bytes:
        n = self._size - self._pos if n is None or n < 0 else min(n, self._size - self._pos)
        out = bytearray()
        while len(out) < max(n, 0):
            index, offset = divmod(self._pos + len(out), self._block)
            out += self._get(index)[offset : offset + n - len(out)]
        self._pos += len(out)
        return bytes(out)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def _get(self, index: int) -> bytes:
        if index not in self._cache:
            start = index * self._block
            self._cache[index] = self._fetch(start, min(start + self._block, self._size) - 1)
        return self._cache[index]


def list_members(fetch: Fetch, size: int) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(RangeFile(fetch, size)) as archive:
        return {info.filename: info for info in archive.infolist()}


def read_member(fetch: Fetch, info: zipfile.ZipInfo) -> bytes:
    """Bytes of one member, fetched with a single range request in the normal case."""
    start = info.header_offset
    raw = fetch(start, start + _LOCAL_HEADER.size + _SLACK + info.compress_size - 1)
    signature, _, flags, method, *_, name_length, extra_length = _LOCAL_HEADER.unpack_from(raw)
    if signature != _LOCAL_SIGNATURE:
        raise ValueError(f"no local header at offset {start} for {info.filename}")
    if flags & 0x1:
        raise ValueError(f"{info.filename} is encrypted")
    begin = _LOCAL_HEADER.size + name_length + extra_length
    if len(raw) < begin + info.compress_size:  # unusually long extra field: fetch the exact range
        raw = fetch(start, start + begin + info.compress_size - 1)
    data = raw[begin : begin + info.compress_size]
    if method == zipfile.ZIP_STORED:
        content = data
    elif method == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(data, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"{info.filename}: unsupported compression method {method}")
    if zlib.crc32(content) != info.CRC:
        raise ValueError(f"{info.filename}: CRC mismatch")
    return content
