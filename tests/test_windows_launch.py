import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import run as cli
from vd import runtime, windows_capture, windows_input
from vd.capture import Frame


def test_windows_cli_selects_dxgi_and_keeps_dry_run_passive(monkeypatch, tmp_path):
    seen = {}

    def fake_capture(**kwargs):
        seen['capture'] = kwargs

    def fake_run_session(**kwargs):
        seen['session'] = kwargs
        kwargs['capture_factory']()
        return {'mode': 'dry-run'}

    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'argv', ['run.py', 'dry-run', '--device-index', '2',
                                    '--output-index', '1', '--recording', str(tmp_path)])
    monkeypatch.setattr(windows_capture, 'WindowsCapture', fake_capture)
    monkeypatch.setattr(runtime, 'run_session', fake_run_session)
    cli.main()

    assert seen['capture'] == {'fps': 60, 'device_idx': 2, 'output_idx': 1}
    assert seen['session']['capture_source'] == 'dxgi'
    assert seen['session']['capture_priority'] is None
    assert seen['session']['physical'] is False


@pytest.mark.parametrize('args', [('--capture-source', 'window'), ('--synthetic',),
                                  ('--capture-priority', '5'), ('--output-index', '-1')])
def test_windows_cli_rejects_unsupported_capture_options(monkeypatch, args):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'argv', ['run.py', 'dry-run', *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2


def test_windows_capture_probe_reports_delivered_rate_and_qpc_age(monkeypatch, capsys):
    class Capture:
        roi = (800, 420, 320, 240)
        physical_roi = (667, 350, 267, 200)
        proc = SimpleNamespace(returncode=0)
        reader = SimpleNamespace(is_alive=lambda: False)

        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def next(self, after=-1, timeout=0):
            media_time = 10 + (after + 1) * .02
            return Frame(after + 1, np.empty((240, 320, 3), np.uint8), media_time,
                         media_time + .004, media_time + .005, 'dxgi_qpc', False,
                         None, 'BGRA', copy_completed_time=media_time + .003,
                         source_size=(1600, 900))

    clock = iter([0, .01, .03, .05, .07])
    monkeypatch.setattr(cli, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'argv', ['run.py', 'capture-probe', '--seconds', '0.06'])
    monkeypatch.setattr(windows_capture, 'WindowsCapture', Capture)
    cli.main()

    report = json.loads(capsys.readouterr().out)
    assert report['frames'] == 3
    assert report['received_fps'] == 50
    assert report['capture_physical_roi'] == [667, 350, 267, 200]
    assert report['mapped_media_age_p95_ms'] == pytest.approx(4)
    assert report['copy_age_median_ms'] == pytest.approx(3)
    assert report['game_render_age_measured'] is False


@pytest.mark.parametrize('physical', [False, True])
def test_dxgi_runtime_only_opens_input_for_run(monkeypatch, tmp_path, physical):
    lifecycle = []

    class Capture:
        cv = threading.Condition()
        proc = SimpleNamespace(poll=lambda: None)
        stopping = False
        error = None
        roi = (800, 420, 320, 240)
        physical_roi = (523, 264, 320, 240)
        alternate_ui_scale = 768 / 1080

        def __enter__(self):
            lifecycle.append('capture')
            return self

        def __exit__(self, *_):
            lifecycle.append('capture closed')

        def next(self, after=-1, timeout=0):
            time.sleep(min(.002, timeout))
            now = time.monotonic()
            self.latest = Frame(after + 1, np.zeros((240, 320, 3), np.uint8),
                                now, now, now, 'dxgi_qpc', False, None, 'BGRA',
                                source_size=(1366, 768))
            return self.latest

        def window_matches_capture(self, hwnd):
            return hwnd == 1

    class Mouse:
        lock = threading.RLock()

        def __init__(self, *, window_matches_capture):
            assert window_matches_capture(1)

        def __enter__(self):
            lifecycle.append('mouse')
            return self

        def __exit__(self, *_):
            lifecycle.append('mouse closed')

        def snapshot(self):
            return SimpleNamespace(held=False, healthy=True)

    class Output:
        error = None

        def __init__(self, mouse):
            assert isinstance(mouse, Mouse)

        def __enter__(self):
            lifecycle.append('output')
            return self

        def __exit__(self, *_):
            lifecycle.append('output closed')

        def pulse(self):
            pytest.fail('No physical output is valid for a blank frame')

    monkeypatch.setattr(windows_input, 'WindowsMouseMonitor', Mouse)
    monkeypatch.setattr(windows_input, 'WindowsSpaceOutput', Output)
    directory = tmp_path / 'session'
    result = runtime.run_session(seconds=.015, synthetic=False, fps=120, directory=directory,
                                 lead_seconds=.004, lead_uncertainty=.020, physical=physical,
                                 capture_source='dxgi', capture_factory=Capture,
                                 capture_priority=None, video_recording=False)

    assert result['frames'] > 0
    assert result['physical_presses'] == 0
    assert lifecycle == (['capture', 'mouse', 'output', 'output closed', 'mouse closed',
                          'capture closed'] if physical else ['capture', 'capture closed'])
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['complete']
    assert manifest['capture_source'] == 'dxgi'
    assert manifest['detector_alternate_ui_scale'] == 768 / 1080
    assert manifest['capture_physical_roi'] == [523, 264, 320, 240]
    assert manifest['held_source'] == ('win32' if physical else 'assumed_for_dry_run')
