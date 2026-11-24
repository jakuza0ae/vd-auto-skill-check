"""Read-only environment report; never create UInput or start capture."""
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def inspect_system():
    import cv2
    import evdev
    checks=[]
    def check(name,ok,detail):
        checks.append(dict(name=name,ok=bool(ok),detail=detail))
    check('wayland',bool(os.environ.get('WAYLAND_DISPLAY')),os.environ.get('WAYLAND_DISPLAY'))
    probe="""import json, gi, dbus, numpy
 gi.require_version('Gst','1.0')
 from gi.repository import Gst
 Gst.init(None)
 print(json.dumps({name:bool(Gst.ElementFactory.find(name)) for name in ['pipewiresrc','videoconvert','appsink','videotestsrc']}))
""".replace('\n ','\n')
    try:
        result=subprocess.run(['/usr/bin/python','-c',probe],capture_output=True,text=True,timeout=10)
        plugins=json.loads(result.stdout) if result.returncode==0 else None
        check('system_capture_dependencies',plugins and all(plugins.values()),
              plugins if plugins else result.stderr.strip())
    except (OSError,subprocess.TimeoutExpired,ValueError) as exc:
        check('system_capture_dependencies',False,str(exc))
    template=Path(__file__).resolve().parents[1]/'assets/space_template.png'
    check('prompt_template',cv2.imread(str(template)) is not None,str(template))
    mice=[]
    errors=[]
    for path in evdev.list_devices():
        device=None
        try:
            device=evdev.InputDevice(path)
            if evdev.ecodes.BTN_LEFT in device.capabilities().get(evdev.ecodes.EV_KEY,[]):
                mice.append(dict(path=path,name=device.name))
        except OSError as exc:
            errors.append(str(exc))
        finally:
            if device is not None:
                device.close()
    check('mouse_read_access',bool(mice),dict(devices=mice,errors=errors))
    check('uinput_write_access',os.access('/dev/uinput',os.W_OK),'/dev/uinput (not opened)')
    versions={name:importlib.metadata.version(name)
              for name in ('numpy','opencv-python-headless','evdev')}
    return dict(platform=platform.platform(),python=platform.python_version(),
                versions=versions,checks=checks,ok=all(c['ok'] for c in checks),
                physical_input_tested=False,portal_selection_tested=False)


def inspect_windows_system():
    checks = []
    versions = {}
    for package in ('numpy', 'opencv-python-headless'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    checks.append(dict(name='python_dependencies', ok=all(versions.values()),
                       detail=versions))

    try:
        from .windows_capture import helper_path
        helper = helper_path()
        checks.append(dict(name='capture_helper', ok=helper.is_file(), detail=str(helper)))
    except (ImportError, OSError) as exc:
        checks.append(dict(name='capture_helper', ok=False, detail=str(exc)))

    for name, libraries in (
        ('dxgi_bindings', {'dxgi': ('CreateDXGIFactory1',),
                           'd3d11': ('D3D11CreateDevice',)}),
        ('win32_input_bindings', {'user32': ('GetAsyncKeyState', 'SendInput')}),
    ):
        try:
            for dll, exports in libraries.items():
                library = ctypes.WinDLL(dll)
                for export in exports:
                    getattr(library, export)
        except (AttributeError, OSError) as exc:
            checks.append(dict(name=name, ok=False, detail=str(exc)))
        else:
            checks.append(dict(name=name, ok=True, detail=list(libraries)))

    try:
        from .vision import Detector
        detector = Detector()
    except (ImportError, OSError, ValueError) as exc:
        checks.append(dict(name='prompt_templates', ok=False, detail=str(exc)))
    else:
        checks.append(dict(name='prompt_templates', ok=True,
                           detail=len(detector.templates)))

    return dict(platform=platform.platform(), python=platform.python_version(),
                versions=versions, checks=checks, ok=all(check['ok'] for check in checks),
                optional_ffmpeg_available=shutil.which('ffmpeg') is not None,
                capture_tested=False, physical_input_tested=False)
