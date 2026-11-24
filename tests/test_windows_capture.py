import ctypes
from ctypes import wintypes
import os
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from vd import windows_capture


_HEADER = struct.Struct('<4sIQQQQIIIIiiii')
_FIXTURE_HELPER = r"""
from pathlib import Path
import os
import sys

for command in sys.stdin.buffer:
    if command == b'failure\n':
        print('Display duplication lost', file=sys.stderr, flush=True)
        sys.exit(1)
    sys.stdout.buffer.write(Path(os.fsdecode(command.removesuffix(b'\n'))).read_bytes())
    sys.stdout.buffer.flush()
"""


@pytest.fixture
def fake_helper(monkeypatch, tmp_path):
    process = subprocess.Popen(
        [sys.executable, '-c', _FIXTURE_HELPER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    monkeypatch.setattr(windows_capture, '_start_helper', lambda **_: process)
    packet_number = 0

    def emit(packet):
        nonlocal packet_number
        packet_number += 1
        path = tmp_path / f'packet-{packet_number}.bin'
        path.write_bytes(packet)
        process.stdin.write(os.fsencode(path) + b'\n')

    try:
        yield process, emit
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def packet(sequence=0, *, source_size=(1920, 1080), roi=(800, 420, 320, 240),
           source_origin=(0, 0), monitor=42, present_ns=None, copy_ns=None):
    now = time.monotonic_ns()
    present_ns = now - 10_000_000 if present_ns is None else present_ns
    copy_ns = now - 7_000_000 if copy_ns is None else copy_ns
    x, y, width, height = roi
    header = _HEADER.pack(
        b'VDD1', 1, sequence, present_ns, copy_ns, monitor, *source_size,
        width, height, x, y, *source_origin,
    )
    return header + bytes((7, 13, 19, 0)) * (width * height)


@pytest.mark.parametrize(
    'source_size,roi',
    [((1920, 1080), (800, 420, 320, 240)),
     ((1600, 900), (667, 350, 267, 200)),
     ((2560, 1440), (1067, 560, 427, 320))],
)
def test_dxgi_roi_is_delivered_with_qpc_timestamp(fake_helper, source_size, roi):
    process, emit = fake_helper
    with windows_capture.WindowsCapture() as capture:
        payload = packet(source_size=source_size, roi=roi, source_origin=(-source_size[0], 0))
        emit(payload)
        frame = capture.next(timeout=4)
        assert frame is not None
        assert frame.image.shape == (240, 320, 3)
        assert tuple(frame.image[0, 0]) == (7, 13, 19)
        assert not frame.image.flags.writeable
        assert frame.source_size == source_size
        assert capture.roi == (800, 420, 320, 240)
        assert capture.physical_roi == roi
        assert frame.timestamp_kind == 'dxgi-desktop-present-qpc'
        assert frame.pts_ns == _HEADER.unpack_from(payload)[3]
        assert frame.media_time == frame.pts_ns / 1e9
        assert frame.media_time <= frame.copy_completed_time <= frame.received_time <= frame.published_time
        assert capture.next(after=frame.sequence, timeout=.01) is None
    assert process.returncode == 0
    assert not capture.reader.is_alive()
    assert not capture._logs.is_alive()


def test_incomplete_new_frame_blocks_prediction_until_publication(fake_helper):
    _, emit = fake_helper
    with windows_capture.WindowsCapture() as capture:
        emit(packet())
        first = capture.next(timeout=4)
        assert first is not None and capture.prediction_current(first)

        second = packet(1)
        emit(second[:_HEADER.size])
        deadline = time.monotonic() + 4
        while not capture.prediction_busy() and time.monotonic() < deadline:
            time.sleep(.001)
        assert capture.prediction_busy()
        assert not capture.prediction_current(first)
        assert capture.next(after=first.sequence, timeout=0) is None

        emit(second[_HEADER.size:])
        newest = capture.next(after=first.sequence, timeout=4)
        assert newest is not None and newest.sequence == 1
        assert newest.source_epoch == first.source_epoch
        assert not capture.prediction_current(first)
        assert capture.prediction_current(newest)


@pytest.mark.parametrize('publish_frame', [False, True])
def test_capture_wait_reserves_last_millisecond_for_dispatch(monkeypatch, publish_frame):
    now = 20.0
    press_at = now + .010
    waits = []
    newest = SimpleNamespace(sequence=1)

    def clock():
        nonlocal now
        current = now
        now += .000025
        return current

    class CoarseCondition:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            # The real frame reader can publish only once this lock is released.
            if publish_frame and waits and press_at - .001 < now < press_at:
                capture.latest = newest

        def wait(self, seconds):
            nonlocal now
            waits.append(seconds)
            now += seconds + .0007

    monkeypatch.setattr(windows_capture, 'time', SimpleNamespace(monotonic=clock))
    capture = windows_capture.WindowsCapture.__new__(windows_capture.WindowsCapture)
    capture.cv = CoarseCondition()
    capture.error = None
    capture.stopping = False
    capture.latest = None
    capture.set_dispatch_deadline(press_at)

    assert capture.next(after=0, timeout=.010) is (newest if publish_frame else None)
    assert waits and max(waits) <= .009
    if publish_frame:
        assert now < press_at
    else:
        assert press_at <= now <= press_at + .0001


@pytest.mark.parametrize('fault', ['changed-display', 'repeated-present', 'helper-died'])
def test_capture_fault_invalidates_published_frame(fake_helper, fault):
    process, emit = fake_helper
    with windows_capture.WindowsCapture() as capture:
        emit(packet())
        first = capture.next(timeout=4)
        assert first is not None
        if fault == 'changed-display':
            emit(packet(1, source_origin=(-1920, 0)))
            error = 'display geometry changed'
        elif fault == 'repeated-present':
            emit(packet(1, present_ns=first.pts_ns))
            error = 'sequence or timestamp'
        else:
            process.stdin.write(b'failure\n')
            error = 'Display duplication lost'
        with pytest.raises(RuntimeError, match=error):
            capture.next(after=first.sequence, timeout=4)
        assert capture.latest is None
        assert not capture.prediction_current(first)
        assert not capture.window_matches_capture(5)


def test_live_window_must_contain_roi_on_same_physical_display(monkeypatch):
    region = windows_capture._DesktopRegion(
        monitor=77, source_size=(1920, 1080), source_origin=(-1920, -100),
        roi=(800, 420, 320, 240),
    )
    windows = windows_capture._WindowGeometry.__new__(windows_capture._WindowGeometry)
    contexts = []
    monkeypatch.setattr(windows, 'set_dpi_context',
                        lambda context: contexts.append(context) or 123, raising=False)
    monkeypatch.setattr(windows, 'monitor_from_window', lambda *_: 77, raising=False)

    def get_info(_monitor, pointer):
        info = ctypes.cast(pointer, ctypes.POINTER(windows_capture._MonitorInfo)).contents
        assert info.size == ctypes.sizeof(windows_capture._MonitorInfo)
        info.monitor = wintypes.RECT(-1920, -100, 0, 980)
        return True

    def get_client(_hwnd, pointer):
        rect = ctypes.cast(pointer, ctypes.POINTER(wintypes.RECT))
        rect[0] = wintypes.RECT(0, 0, 320, 240)
        return True

    def move_client(_hwnd, pointer):
        pos = ctypes.cast(pointer, ctypes.POINTER(wintypes.POINT))
        pos[0] = wintypes.POINT(-1120, 320)
        return True

    monkeypatch.setattr(windows, 'get_monitor_info', get_info, raising=False)
    monkeypatch.setattr(windows, 'get_client_rect', get_client, raising=False)
    monkeypatch.setattr(windows, 'client_to_screen', move_client, raising=False)
    assert windows.contains(5, region)
    assert contexts.pop() == 123

    monkeypatch.setattr(windows, 'monitor_from_window', lambda *_: 78)
    assert not windows.contains(5, region)
    assert contexts.pop() == 123

    monkeypatch.setattr(windows, 'monitor_from_window', lambda *_: 77)
    shifted_region = windows_capture._DesktopRegion(
        region.monitor, region.source_size, (-1920, 0), region.roi,
    )
    assert not windows.contains(5, shifted_region)

    clipped_region = windows_capture._DesktopRegion(
        region.monitor, region.source_size, region.source_origin, (801, 420, 320, 240),
    )
    assert not windows.contains(5, clipped_region)
