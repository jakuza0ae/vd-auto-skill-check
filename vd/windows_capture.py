"""Capture a Windows display ROI through the native DXGI Desktop Duplication helper."""
from __future__ import annotations

from collections import deque
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from vd.capture import DEFAULT_ROI, Frame


_FRAME_HEADER = struct.Struct('<4sIQQQQIIIIiiii')
_FRAME_TIMING = struct.Struct('<QI4x')
_MAX_ROI_BYTES = 64 * 1024 * 1024
_TIMESTAMP_KIND = 'dxgi-desktop-present-qpc'
_PER_MONITOR_AWARE_V2 = -4
_DISPATCH_WAIT_GUARD = .017


def helper_path():
    return (Path(__file__).resolve().parents[1] / 'build' / 'windows-capture'
            / 'vd-windows-capture.exe')


def _start_helper(*, fps, device_idx, output_idx):
    if sys.platform != 'win32':
        raise RuntimeError('DXGI capture requires Windows')
    executable = helper_path()
    if not executable.is_file():
        raise RuntimeError(f'Windows capture helper is missing. Build it first: {executable}')

    command = [str(executable), '--fps', str(fps)]
    if device_idx is not None:
        command.extend(('--device-index', str(device_idx)))
    if output_idx is not None:
        command.extend(('--output-index', str(output_idx)))
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    except OSError as exc:
        raise RuntimeError(f'Cannot start the Windows capture helper: {exc}') from exc


def _read_exact(stream, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError('Windows capture helper closed the frame stream')
        offset += count
    return data


@dataclass(frozen=True)
class _DesktopRegion:
    monitor: int
    source_size: tuple[int, int]
    source_origin: tuple[int, int]
    roi: tuple[int, int, int, int]

    @property
    def screen_bounds(self):
        x, y = self.source_origin
        width, height = self.source_size
        return x, y, x + width, y + height

    @property
    def screen_roi(self):
        x, y, width, height = self.roi
        left, top = self.source_origin
        return left + x, top + y, left + x + width, top + y + height


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ('size', wintypes.DWORD),
        ('monitor', wintypes.RECT),
        ('work_area', wintypes.RECT),
        ('flags', wintypes.DWORD),
    ]


class _WindowGeometry:
    """Check ROI containment in physical screen pixels on a mixed-DPI desktop."""

    def __init__(self):
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.set_dpi_context = user32.SetThreadDpiAwarenessContext
        self.set_dpi_context.argtypes = (wintypes.HANDLE,)
        self.set_dpi_context.restype = wintypes.HANDLE
        self.monitor_from_window = user32.MonitorFromWindow
        self.monitor_from_window.argtypes = (wintypes.HWND, wintypes.DWORD)
        self.monitor_from_window.restype = wintypes.HMONITOR
        self.get_monitor_info = user32.GetMonitorInfoW
        self.get_monitor_info.argtypes = (wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo))
        self.get_monitor_info.restype = wintypes.BOOL
        self.get_client_rect = user32.GetClientRect
        self.get_client_rect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
        self.get_client_rect.restype = wintypes.BOOL
        self.client_to_screen = user32.ClientToScreen
        self.client_to_screen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
        self.client_to_screen.restype = wintypes.BOOL

    def contains(self, hwnd, region):
        previous_context = self.set_dpi_context(ctypes.c_void_p(_PER_MONITOR_AWARE_V2))
        if not previous_context:
            return False
        try:
            if self.monitor_from_window(hwnd, 0) != region.monitor:
                return False
            monitor = _MonitorInfo(size=ctypes.sizeof(_MonitorInfo))
            if not self.get_monitor_info(region.monitor, ctypes.byref(monitor)):
                return False
            bounds = monitor.monitor
            if (bounds.left, bounds.top, bounds.right, bounds.bottom) != region.screen_bounds:
                return False
            window = wintypes.RECT()
            if not self.get_client_rect(hwnd, ctypes.byref(window)):
                return False
            origin = wintypes.POINT(window.left, window.top)
            if not self.client_to_screen(hwnd, ctypes.byref(origin)):
                return False
            left, top, right, bottom = region.screen_roi
            return (origin.x <= left < right <= origin.x + window.right - window.left
                    and origin.y <= top < bottom <= origin.y + window.bottom - window.top)
        finally:
            self.set_dpi_context(previous_context)


class WindowsCapture:
    """Run one DXGI helper and expose only the newest immutable BGR frame."""

    def __init__(self, *, fps=60, device_idx=None, output_idx=None):
        if type(fps) is not int or not 1 <= fps <= 240:
            raise ValueError('FPS must be an integer in [1, 240]')
        for index in (device_idx, output_idx):
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError('DXGI adapter and display indices must be non-negative integers')

        self.roi = DEFAULT_ROI
        self.cv = threading.Condition()
        self.latest = None
        self.error = None
        self.stopping = False
        self._pending_frame = False
        self._dispatch_deadline = None
        self._region = None
        self._window_geometry = None
        self._stderr = deque(maxlen=15)
        self.proc = _start_helper(fps=fps, device_idx=device_idx, output_idx=output_idx)
        self._source_epoch = self.proc.pid, time.monotonic_ns()
        self._logs = threading.Thread(target=self._read_logs, name='VD-DXGI-log', daemon=True)
        self.reader = threading.Thread(target=self._read, name='VD-DXGI-reader', daemon=True)
        try:
            self._logs.start()
            self.reader.start()
        except BaseException:
            self.close()
            raise

    def _read_logs(self):
        for line in self.proc.stderr:
            self._stderr.append(line.decode(errors='replace').rstrip())

    def _read(self):
        previous_sequence = -1
        previous_present_ns = 0
        target_width, target_height = self.roi[2:]
        try:
            while True:
                if self.stopping:
                    # Keep the pipe writable until the helper observes stdin EOF.
                    while self.proc.stdout.read(65536):
                        pass
                    return
                header = _read_exact(self.proc.stdout, _FRAME_HEADER.size)
                with self.cv:
                    self._pending_frame = True
                (magic, version, sequence, present_ns, copy_ns, monitor,
                 width, height, roi_width, roi_height, roi_x, roi_y,
                 source_x, source_y) = _FRAME_HEADER.unpack(header)
                if magic != b'VDD1' or version not in (1, 2):
                    raise ValueError('Unsupported Windows capture frame protocol')
                size = roi_width * roi_height * 4
                if (monitor == 0 or roi_x < 0 or roi_y < 0 or not 0 < size <= _MAX_ROI_BYTES
                        or roi_x + roi_width > width or roi_y + roi_height > height):
                    raise ValueError('Invalid Windows capture desktop or ROI geometry')
                if sequence <= previous_sequence or not previous_present_ns < present_ns <= copy_ns:
                    raise ValueError('Invalid Windows capture frame sequence or timestamp')

                acquired_ns = accumulated_frames = None
                if version == 2:
                    timing = _read_exact(self.proc.stdout, _FRAME_TIMING.size)
                    acquired_ns, accumulated_frames = _FRAME_TIMING.unpack(timing)
                    if not present_ns <= acquired_ns <= copy_ns:
                        raise ValueError('Invalid Windows capture acquisition timestamp')

                region = _DesktopRegion(monitor, (width, height), (source_x, source_y),
                                        (roi_x, roi_y, roi_width, roi_height))
                with self.cv:
                    if self._region is not None and region != self._region:
                        raise ValueError('Windows capture display geometry changed. Restart capture.')

                pixels = _read_exact(self.proc.stdout, size)
                received_ns = time.monotonic_ns()
                if copy_ns > received_ns + 2_000_000:
                    raise ValueError('Windows capture and Python monotonic timestamps disagree')
                bgra = np.frombuffer(pixels, dtype=np.uint8).reshape(roi_height, roi_width, 4)
                image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                if (roi_width, roi_height) != (target_width, target_height):
                    image = cv2.resize(image, (target_width, target_height),
                                       interpolation=cv2.INTER_LINEAR)
                image.flags.writeable = False
                previous_sequence, previous_present_ns = sequence, present_ns
                with self.cv:
                    if self.stopping:
                        continue
                    published = time.monotonic()
                    self._region = region
                    self.latest = Frame(
                        sequence=sequence, image=image, media_time=present_ns / 1e9,
                        received_time=received_ns / 1e9, consumed_time=published,
                        timestamp_kind=_TIMESTAMP_KIND, synthetic=False, pts_ns=present_ns,
                        negotiated_caps=(f'dxgi-bgra;raw-roi={roi_width}x{roi_height};'
                                         f'bgr={target_width}x{target_height}'),
                        source_size=region.source_size,
                        source_generation=1, copy_completed_time=copy_ns / 1e9,
                        published_time=published, source_epoch=self._source_epoch,
                        source_acquired_time=acquired_ns / 1e9 if acquired_ns is not None else None,
                        source_accumulated_frames=accumulated_frames,
                    )
                    self._pending_frame = False
                    self.cv.notify_all()
        except Exception as exc:
            # Let the sibling pipe drain terminal diagnostics before reporting EOF.
            if isinstance(exc, EOFError):
                with self.cv:
                    self.latest = None
                self._logs.join(timeout=.1)
            with self.cv:
                if not self.stopping:
                    detail = ' | '.join(self._stderr.copy())
                    self.error = f'{exc}: {detail}' if detail else str(exc)
                    self.latest = None
                self._pending_frame = False
                self.cv.notify_all()

    def set_dispatch_deadline(self, deadline):
        self._dispatch_deadline = deadline

    def next(self, after=-1, timeout=1.0):
        deadline = time.monotonic() + timeout
        poll_window = 0.0
        dispatch_deadline = self._dispatch_deadline
        if dispatch_deadline is not None and dispatch_deadline <= deadline:
            deadline = dispatch_deadline
            poll_window = _DISPATCH_WAIT_GUARD
        while True:
            with self.cv:
                if self.stopping:
                    return None
                if self.error:
                    raise RuntimeError(self.error)
                if self.latest is not None and self.latest.sequence > after:
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                if remaining > poll_window:
                    self.cv.wait(remaining - poll_window)
                    continue
            # Condition.wait uses a millisecond Windows lock wait. Use CPython's
            # high-resolution sleep close to a deadline, outside the capture lock.
            time.sleep(min(.001, max(0.0, remaining - .0005)))

    def prediction_current(self, frame):
        with self.cv:
            return frame is not None and self.latest is frame and not self._pending_frame

    def prediction_busy(self):
        with self.cv:
            return self._pending_frame

    def window_matches_capture(self, hwnd):
        with self.cv:
            region = (self._region if self.latest is not None and not self.stopping
                      and not self.error and self.proc.poll() is None else None)
        if region is None or not hwnd:
            return False
        if self._window_geometry is None:
            self._window_geometry = _WindowGeometry()
        return self._window_geometry.contains(hwnd, region)

    @property
    def physical_roi(self):
        with self.cv:
            return self._region.roi if self._region is not None else None

    @property
    def alternate_ui_scale(self):
        with self.cv:
            region = self._region
        if region is not None and region.roi[2:] == self.roi[2:]:
            width, height = region.source_size
            scale = min(width / 1920, height / 1080)
            if scale < 1:
                return scale
        return None

    def close(self, *, fast=False):
        with self.cv:
            if self.stopping:
                return
            self.stopping = True
            self.latest = None
            self.cv.notify_all()
        timeout = 1.2 if fast else 2
        self.proc.stdin.close()
        try:
            if self.error and self.proc.poll() is None:
                self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        finally:
            self._close_readers(timeout)

    def _close_readers(self, timeout):
        for thread in (self.reader, self._logs):
            if thread.ident is not None:
                thread.join(timeout=timeout)
        self.proc.stdout.close()
        self.proc.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        self.close(fast=exc_type is not None and issubclass(exc_type, KeyboardInterrupt))
