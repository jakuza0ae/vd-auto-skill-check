import ctypes
import os
from pathlib import PureWindowsPath
import threading
from types import SimpleNamespace

import pytest

from vd import windows_input
from vd.dispatch import dispatch
from vd.engine import Engine
from vd.input_state import InputDisarmed
from vd.motion import Motion
from vd.planning import Arc


class FakeWin32:
    def __init__(self):
        self.target = (42, 50)
        self.process_path = PureWindowsPath(r'C:\Roblox\RobloxPlayerBeta.exe')
        self.lmb = True
        self.space = False
        self.writes = []
        self.release_failures = 0

    def key_down(self, key):
        return {0x01: self.lmb, 0x20: self.space}[key]

    def foreground(self):
        return self.target

    def image_path(self, pid):
        assert pid == self.target[1]
        return self.process_path

    def set_space(self, down):
        self.writes.append(down)
        if not down and self.release_failures:
            self.release_failures -= 1
            raise OSError('keyup failed')


class FakeTimer:
    def __init__(self, interval, release):
        self.interval = interval
        self.release = release
        self.cancelled = False
        self.started = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def input_devices(monkeypatch):
    monkeypatch.setattr(windows_input.threading, 'Timer', FakeTimer)
    api = FakeWin32()
    visible_windows = {42}
    with windows_input.WindowsMouseMonitor(
        window_matches_capture=visible_windows.__contains__, api=api,
    ) as mouse, windows_input.WindowsSpaceOutput(mouse) as output:
        yield api, visible_windows, mouse, output


def test_mouse_tracks_physical_button_and_latches_focus_loss(input_devices):
    api, visible_windows, mouse, _ = input_devices
    visible_windows.add(43)
    assert mouse.snapshot().held
    api.target = (43, 51)
    assert not mouse.snapshot().held
    api.target = (42, 50)
    assert not mouse.snapshot().held
    api.lmb = False
    assert not mouse.snapshot().held
    api.lmb = True
    assert mouse.snapshot().held
    mouse.close()
    assert not mouse.snapshot().healthy


@pytest.mark.parametrize('path', [None, r'C:\Browser\chrome.exe',
                                  r'C:\Roblox\RobloxStudioBeta.exe',
                                  r'C:\Other\Windows10Universal.exe'])
def test_mouse_does_not_arm_outside_roblox(input_devices, path):
    api, _, mouse, output = input_devices
    api.process_path = PureWindowsPath(path) if path is not None else None
    assert not mouse.snapshot().held
    with pytest.raises(InputDisarmed):
        output.pulse()
    assert api.writes == []


@pytest.mark.parametrize('change', ['focus', 'pid', 'capture', 'lmb', 'space'])
def test_space_rechecks_target_and_buttons_before_keydown(input_devices, change):
    api, visible_windows, mouse, output = input_devices
    assert mouse.snapshot().held
    if change == 'focus':
        api.target = None
    elif change == 'pid':
        api.target = (42, 51)
    elif change == 'capture':
        visible_windows.clear()
    elif change == 'lmb':
        api.lmb = False
    else:
        api.space = True
    with pytest.raises(InputDisarmed):
        output.pulse()
    assert api.writes == []


def test_space_releases_after_focus_change(input_devices):
    api, _, mouse, output = input_devices
    assert mouse.snapshot().held
    result = output.pulse()
    release_timer = output._release_timer
    assert result.syn_completed_at >= result.requested_at
    assert release_timer.started
    assert release_timer.interval == .080
    assert api.writes == [True]
    api.target = None
    release_timer.release()
    output.close()
    assert api.writes == [True, False]


def test_space_overlap_disarms_without_releasing_the_current_pulse(input_devices):
    api, _, _, output = input_devices
    output.pulse()
    timer = output._release_timer

    with pytest.raises(InputDisarmed, match='not been released') as overlap:
        output.pulse()

    assert isinstance(overlap.value, RuntimeError)
    assert output._release_timer is timer
    assert not timer.cancelled
    assert api.writes == [True]
    timer.release()
    assert api.writes == [True, False]


def test_cleanup_retries_failed_release(input_devices):
    api, _, _, output = input_devices
    output.pulse()
    api.release_failures = 1
    output._release_timer.release()
    assert output.error == 'keyup failed'
    output.close()
    assert api.writes == [True, False, False]


def test_cleanup_cancels_timer_and_releases_space_once(input_devices):
    api, _, _, output = input_devices
    output.pulse()
    timer = output._release_timer
    output.close()
    assert timer.cancelled
    timer.release()
    output.close()
    assert api.writes == [True, False]


def test_cleanup_releases_after_interrupted_keydown(input_devices, monkeypatch):
    api, _, mouse, output = input_devices

    def interrupted(down):
        api.writes.append(down)
        if down:
            raise KeyboardInterrupt

    monkeypatch.setattr(api, 'set_space', interrupted)
    with pytest.raises(KeyboardInterrupt), output:
        output.pulse()
    assert api.writes == [True, False]
    assert not output.error


def test_windows_button_state_uses_only_high_bit():
    api = windows_input._Win32.__new__(windows_input._Win32)
    api.async_key_state = lambda key: 1
    assert not api.key_down(0x01)
    api.async_key_state = lambda key: -32768
    assert api.key_down(0x01)


@pytest.mark.skipif(not hasattr(ctypes, 'WinDLL'), reason='requires Win32')
def test_native_win32_api_opens_current_process_without_input():
    path = windows_input._Win32().image_path(os.getpid())
    assert path is not None and path.suffix.casefold() == '.exe'


@pytest.mark.parametrize('success', [False, True])
def test_windows_closes_process_handle_after_query(success):
    api = windows_input._Win32.__new__(windows_input._Win32)
    closed = []
    path = PureWindowsPath(r'C:\Roblox\RobloxPlayerBeta.exe')
    api.open_process = lambda access, inherit, pid: 99
    api.close_handle = closed.append

    def query_image(handle, flags, buffer, size):
        assert handle == 99
        if success:
            buffer.value = str(path)
        return success

    api.process_image = query_image
    assert api.image_path(1) == (path if success else None)
    assert closed == [99]


def test_send_input_encodes_native_space_events(monkeypatch):
    events = []
    api = windows_input._Win32.__new__(windows_input._Win32)
    monkeypatch.setattr(ctypes, 'set_last_error', lambda _: None, raising=False)

    def send_input(count, pointer, size):
        assert count == 1
        assert size == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        event = ctypes.cast(pointer, ctypes.POINTER(windows_input._Input)).contents
        events.append((event.type, event.payload.ki.wVk, event.payload.ki.wScan,
                       event.payload.ki.dwFlags))
        return count

    api.send_input = send_input
    api.set_space(True)
    api.set_space(False)
    assert events == [(1, 0, 0x39, 0x0008), (1, 0, 0x39, 0x000a)]
    api.send_input = lambda *args: 0
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: 0, raising=False)
    with pytest.raises(OSError, match='keydown was rejected by Windows'):
        api.set_space(True)


def test_dispatch_cancels_when_win32_focus_changes_after_claim(input_devices, monkeypatch):
    api, _, mouse, output = input_devices
    engine = Engine(lead_seconds=.1, lead_uncertainty=0)
    engine.planner.begin(Arc(96, 8), 60)
    motion = Motion(1.0, 70, 300, 0, 0, 1.0, 6)
    assert engine.planner.update(motion, frame_at=1.0, now=1.0) is not None
    frame = SimpleNamespace(sequence=7, synthetic=False, media_time=1.0)
    capture = SimpleNamespace(cv=threading.Condition(), stopping=False, error=None,
                              latest=frame, proc=SimpleNamespace(poll=lambda: None))
    pulse = output.pulse

    def lose_focus():
        api.target = None
        return pulse()

    monkeypatch.setattr(output, 'pulse', lose_focus)
    result = dispatch(engine, capture, frame.sequence, mouse=mouse, output=output, clock=lambda: 1.0)
    assert result is None
    assert engine.reason == 'INPUT_DISARMED'
    assert [event['kind'] for event in engine.take_events()] == ['PRESS_CLAIM', 'INPUT_DISARMED']
    assert api.writes == []
