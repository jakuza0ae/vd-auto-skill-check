"""Foreground-gated Windows mouse state and Space pulses via Win32."""

import ctypes
from pathlib import PureWindowsPath
import threading
import time

from .input_state import InputDisarmed, Keydown, MouseState


_VK_LBUTTON = 0x01
_VK_SPACE = 0x20
_SCAN_SPACE = 0x39
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_DWORD = ctypes.c_uint32
_WORD = ctypes.c_uint16
_LONG = ctypes.c_int32
_ULONG_PTR = ctypes.c_size_t
_HANDLE = ctypes.c_void_p


class _MouseInput(ctypes.Structure):
    _fields_ = [('dx', _LONG), ('dy', _LONG), ('mouseData', _DWORD),
                ('dwFlags', _DWORD), ('time', _DWORD), ('dwExtraInfo', _ULONG_PTR)]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [('wVk', _WORD), ('wScan', _WORD), ('dwFlags', _DWORD),
                ('time', _DWORD), ('dwExtraInfo', _ULONG_PTR)]


class _HardwareInput(ctypes.Structure):
    _fields_ = [('uMsg', _DWORD), ('wParamL', _WORD), ('wParamH', _WORD)]


class _InputPayload(ctypes.Union):
    _fields_ = [('mi', _MouseInput), ('ki', _KeyboardInput), ('hi', _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [('type', _DWORD), ('payload', _InputPayload)]


def _function(dll, name, result, *arguments):
    function = getattr(dll, name)
    function.argtypes = arguments
    function.restype = result
    return function


class _Win32:
    def __init__(self):
        if not hasattr(ctypes, 'WinDLL'):
            raise RuntimeError('Win32 input requires Windows')
        user = ctypes.WinDLL('user32', use_last_error=True)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.async_key_state = _function(user, 'GetAsyncKeyState', ctypes.c_int16, ctypes.c_int)
        self.foreground_window = _function(user, 'GetForegroundWindow', _HANDLE)
        self.window_process = _function(user, 'GetWindowThreadProcessId', _DWORD,
                                        _HANDLE, ctypes.POINTER(_DWORD))
        self.send_input = _function(user, 'SendInput', _DWORD,
                                    _DWORD, ctypes.POINTER(_Input), ctypes.c_int)
        self.open_process = _function(kernel, 'OpenProcess', _HANDLE, _DWORD, _LONG, _DWORD)
        self.process_image = _function(kernel, 'QueryFullProcessImageNameW', _LONG,
                                        _HANDLE, _DWORD, ctypes.c_wchar_p, ctypes.POINTER(_DWORD))
        self.close_handle = _function(kernel, 'CloseHandle', _LONG, _HANDLE)

    def key_down(self, key):
        return bool(self.async_key_state(key) & 0x8000)

    def foreground(self):
        hwnd = self.foreground_window()
        pid = _DWORD()
        if hwnd and self.window_process(hwnd, ctypes.byref(pid)) and pid.value:
            return hwnd, pid.value
        return None

    def image_path(self, pid):
        process = self.open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not process:
            return None
        try:
            size = _DWORD(32768)
            path = ctypes.create_unicode_buffer(size.value)
            if self.process_image(process, 0, path, ctypes.byref(size)):
                return PureWindowsPath(path.value)
            return None
        finally:
            self.close_handle(process)

    def set_space(self, down):
        event = _Input(type=_INPUT_KEYBOARD)
        event.payload.ki.wScan = _SCAN_SPACE
        event.payload.ki.dwFlags = _KEYEVENTF_SCANCODE | (0 if down else _KEYEVENTF_KEYUP)
        ctypes.set_last_error(0)
        if self.send_input(1, ctypes.byref(event), ctypes.sizeof(event)) != 1:
            error = ctypes.get_last_error()
            action = 'keydown' if down else 'keyup'
            if error:
                message = ctypes.FormatError(error).strip()
                raise OSError(error, f'SendInput Space {action} failed: {message}')
            raise OSError(f'SendInput Space {action} was rejected by Windows')


def _is_roblox_player(path):
    if path is None:
        return False
    name = path.name.casefold()
    return name == 'robloxplayerbeta.exe' or (
        name == 'windows10universal.exe'
        and path.parent.name.casefold().startswith('robloxcorporation.roblox_')
    )


class WindowsMouseMonitor:
    def __init__(self, *, window_matches_capture, api=None):
        self.lock = threading.RLock()
        self._api = _Win32() if api is None else api
        self._window_matches_capture = window_matches_capture
        self._closed = False
        self._lmb_held = False
        self._target = None

    def _matching_target(self):
        target = self._api.foreground()
        if target is None or not self._window_matches_capture(target[0]):
            return None
        return target

    def snapshot(self):
        with self.lock:
            if self._closed:
                return MouseState(False, False, None, 'Windows mouse monitor is closed')
            held = self._api.key_down(_VK_LBUTTON)
            if not held:
                self._target = None
            elif not self._lmb_held:
                target = self._matching_target()
                if target is not None and _is_roblox_player(self._api.image_path(target[1])):
                    self._target = target
            elif self._target is not None and self._matching_target() != self._target:
                # A different window cannot inherit an already held button.
                self._target = None
            self._lmb_held = held
            return MouseState(self._target is not None, True, 'win32', None)

    def _check_keydown(self):
        if self._api.key_down(_VK_SPACE):
            raise InputDisarmed('Space is already down')
        if (not self.snapshot().held or self._api.foreground() != self._target
                or not self._api.key_down(_VK_LBUTTON)):
            self._target = None
            raise InputDisarmed('Roblox lost focus, moved outside capture, or LMB was released')

    def close(self):
        with self.lock:
            self._closed = True
            self._target = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class WindowsSpaceOutput:
    def __init__(self, mouse, pulse_seconds=.080):
        if not 0.01 <= pulse_seconds <= 0.1:
            raise ValueError('Space pulse must be in [10, 100] ms')
        self._mouse = mouse
        self._api = mouse._api
        self._pulse_seconds = pulse_seconds
        self._lock = threading.Lock()
        self._closed = False
        self._pressed = False
        self._release_timer = None
        self.error = None

    def _release(self):
        with self._lock:
            self._release_timer = None
            if not self._pressed or self._closed:
                return
            try:
                self._api.set_space(False)
                self._pressed = False
            except OSError as exc:
                self.error = str(exc)

    def pulse(self):
        with self._mouse.lock, self._lock:
            if self._closed:
                raise RuntimeError('Space output is closed')
            if self._pressed:
                raise InputDisarmed('Previous Space pulse has not been released')
            self._mouse._check_keydown()
            requested = time.monotonic()
            self._pressed = True
            try:
                self._api.set_space(True)
            except OSError:
                self._pressed = False
                raise
            completed = time.monotonic()
            self._release_timer = threading.Timer(self._pulse_seconds, self._release)
            self._release_timer.daemon = True
            self._release_timer.start()
            return Keydown(requested, completed)

    def close(self):
        with self._lock:
            self._closed = True
            if self._release_timer is not None:
                self._release_timer.cancel()
                self._release_timer = None
            if self._pressed:
                try:
                    self._api.set_space(False)
                    self._pressed = False
                except OSError as exc:
                    self.error = str(exc)
                    raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
