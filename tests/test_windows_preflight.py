from types import SimpleNamespace

import pytest

from vd import preflight
from vd import windows_capture


@pytest.mark.parametrize('dependencies_ok', [True, False])
def test_windows_preflight_checks_bindings_without_calling_them(
    monkeypatch, tmp_path, dependencies_ok,
):
    def version(package):
        if package == 'numpy' and not dependencies_ok:
            raise preflight.importlib.metadata.PackageNotFoundError(package)
        return 'test-version'

    def unused_api():
        pytest.fail('Preflight must not start capture or send input')

    def load_library(name):
        exports = {
            'dxgi': SimpleNamespace(CreateDXGIFactory1=unused_api),
            'd3d11': SimpleNamespace(D3D11CreateDevice=unused_api),
            'user32': SimpleNamespace(GetAsyncKeyState=unused_api, SendInput=unused_api),
        }
        if name == 'd3d11' and not dependencies_ok:
            raise OSError('D3D11 unavailable')
        return exports[name]

    monkeypatch.setattr(preflight.importlib.metadata, 'version', version)
    monkeypatch.setattr(preflight.ctypes, 'WinDLL', load_library, raising=False)
    monkeypatch.setattr(preflight.shutil, 'which', lambda tool: None)
    helper = tmp_path / 'vd-windows-capture.exe'
    if dependencies_ok:
        helper.touch()
    monkeypatch.setattr(windows_capture, 'helper_path', lambda: helper)

    result = preflight.inspect_windows_system()

    assert result['ok'] is dependencies_ok
    assert {row['name']: row['ok'] for row in result['checks']} == {
        'python_dependencies': dependencies_ok, 'capture_helper': dependencies_ok,
        'dxgi_bindings': dependencies_ok,
        'win32_input_bindings': True, 'prompt_templates': True,
    }
    assert result['optional_ffmpeg_available'] is False
    assert result['capture_tested'] is False
    assert result['physical_input_tested'] is False
