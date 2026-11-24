#!/usr/bin/env python3
"""Development entry point for the independent VD rewrite."""
import os
import sys
from pathlib import Path

if __name__=='__main__' and sys.platform!='win32':
    venv=Path(__file__).resolve().parent/'.venv'
    python=venv/'bin/python'
    if python.exists() and Path(sys.prefix).resolve()!=venv.resolve():
        os.execv(str(python),[str(python),str(Path(__file__).resolve()),*sys.argv[1:]])

import argparse
from datetime import datetime
from functools import partial
import json
import statistics
import time

VARIANTS = {
    # FPS, minimum nice value for the capture worker (higher yields more CPU).
    'baseline': (60, 5),
    'responsive': (60, 0),
    'quiet': (60, 10),
    'fps50': (50, 5),
    'fps45': (45, 5),
    'fps30': (30, 5),
    'deep-quiet': (30, 10),
}


def main():
    parser = argparse.ArgumentParser(description='VD rewrite: capture, dry-run, replay')
    parser.add_argument('mode', choices=['capture-probe','dry-run','run','replay','summary','preflight'])
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--variant', choices=VARIANTS,
                        help='capture profile; explicit --fps/--capture-priority override it')
    parser.add_argument('--capture-source', choices=('monitor', 'window', 'region'), default='monitor',
                        help='capture a monitor, game window, or KWin region around the screen center')
    parser.add_argument('--normalize-window-scale', action='store_true',
                        help='crop a proportional center region from a smaller 16:9 game window and resize it for the solver')
    parser.add_argument('--fps', type=int, default=None,
                        help='maximum capture FPS; synthetic capture uses fixed FPS')
    parser.add_argument('--capture-priority', type=int, choices=range(20), default=None,
                        help='Linux capture worker nice value (0 is more CPU, 19 less)')
    parser.add_argument('--device-index', type=int,
                        help='Windows DXGI graphics adapter index, starting at zero')
    parser.add_argument('--output-index', type=int,
                        help='Windows DXGI display index on the selected adapter, starting at zero')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--recording', type=Path, help='New output directory, or input for replay')
    parser.add_argument('--no-recording', action='store_true',
                        help='Disable video/decision recording to reduce CPU and disk load')
    video = parser.add_mutually_exclusive_group()
    video.add_argument('--video', dest='video_recording', action='store_true',
                       help='Record frames as well as decision events for later replay')
    video.add_argument('--no-video', dest='video_recording', action='store_false',
                       help='Keep decision events and performance stats but skip frame copies and video encoding')
    parser.set_defaults(video_recording=False)
    parser.add_argument('--lead-ms', '--response-delay-ms', type=float)
    parser.add_argument('--lead-uncertainty-ms', '--response-uncertainty-ms', type=float, default=20)
    parser.add_argument('--learn-lead', action=argparse.BooleanOptionalAction, default=None,
                        help='Learn dispatch delay only; configured game response delay remains fixed (default in run)')
    args = parser.parse_args()
    windows=sys.platform=='win32'
    if args.mode=='summary':
        if args.recording is None:
            parser.error('summary requires --recording')
        from vd.recording import summarize_recording
        print(json.dumps(summarize_recording(args.recording),indent=2,ensure_ascii=windows))
        return
    if args.mode=='preflight':
        if windows:
            from vd.preflight import inspect_windows_system
            result=inspect_windows_system()
        else:
            from vd.preflight import inspect_system
            result=inspect_system()
        print(json.dumps(result,indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',
                                   encoding='utf-8')
        if not result['ok']:
            raise SystemExit(1)
        return
    if args.seconds is None and args.mode!='run':
        args.seconds=5
    if args.seconds is not None and not 0 < args.seconds <= 86400:
        parser.error('seconds must be in (0, 86400]')
    if args.mode=='run' and args.synthetic:
        parser.error('run does not accept synthetic capture')
    if args.mode=='run' and args.no_recording:
        parser.error('run always records; --no-recording is only for dry-run')
    if args.normalize_window_scale and args.capture_source != 'window':
        parser.error('--normalize-window-scale requires --capture-source window')
    if args.learn_lead is True and args.mode!='run':
        parser.error('--learn-lead requires run')
    windows_capture=windows and args.mode in ('capture-probe', 'dry-run', 'run')
    if windows_capture:
        if args.synthetic or args.capture_source!='monitor' or args.capture_priority is not None:
            parser.error('Windows DXGI capture supports monitor capture; synthetic sources, '
                         'window/region capture and worker nice values require Linux')
        if any(index is not None and index<0 for index in (args.device_index,args.output_index)):
            parser.error('DXGI adapter and display indices must be non-negative')
        args.variant=args.variant or 'quiet'
        profile_fps=VARIANTS[args.variant][0]
    else:
        if args.device_index is not None or args.output_index is not None:
            parser.error('DXGI adapter and display indices require Windows capture')
        args.variant=args.variant or 'quiet'
        profile_fps, profile_priority=VARIANTS[args.variant]
        if args.capture_priority is None:
            args.capture_priority=profile_priority
    args.fps=profile_fps if args.fps is None else args.fps
    if not 1<=args.fps<=240:
        parser.error('fps must be in [1, 240]')
    if args.lead_ms is None:
        from vd.planning import DEFAULT_RESPONSE_DELAY_SECONDS
        args.lead_ms=DEFAULT_RESPONSE_DELAY_SECONDS*1000
    if not 0<=args.lead_ms<=300 or not 0<=args.lead_uncertainty_ms<=100:
        parser.error('lead must be in [0, 300] ms; uncertainty in [0, 100] ms')
    capture_source=args.capture_source
    session_options={}
    if windows_capture:
        from vd.windows_capture import WindowsCapture
        capture_source='dxgi'
        capture_factory=partial(WindowsCapture,fps=args.fps,device_idx=args.device_index,
                                output_idx=args.output_index)
        session_options['capture_factory']=capture_factory
    if args.mode!='capture-probe':
        from vd.runtime import run_session, replay
        if args.mode=='replay':
            if args.recording is None:
                parser.error('replay requires --recording')
            result=replay(args.recording)
        else:
            directory=args.recording or (Path(__file__).resolve().parent/'recordings'/
                         datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            if args.mode=='run':
                if windows_capture:
                    print('RUN: Hold LMB in the foreground Roblox window. Ctrl+C to stop. '
                          f'DXGI cap={args.fps} FPS, response={args.lead_ms:g} ms, '
                          f'dispatch learning={args.learn_lead is not False}, '
                          f'video={args.video_recording}.',flush=True)
                else:
                    print(f'RUN: Space при удержании LMB; Ctrl+C для остановки; '
                          f'profile={args.variant} capture={args.fps} FPS/'
                          f'nice≥{args.capture_priority}; source={capture_source}; '
                          f'geometry={"source-region" if args.capture_source == "region" else "scaled-after-capture" if args.normalize_window_scale else "fixed"}; '
                          f'lead={args.lead_ms:g} ms '
                          f'(обучение задержки отправки: {"включена" if args.learn_lead is not False else "только наблюдение"}; '
                          f'видеозапись: {"вкл." if args.video_recording else "выкл."}; '
                          f'журнал событий: вкл.).',flush=True)
                try:
                    result=run_session(seconds=args.seconds,synthetic=args.synthetic,fps=args.fps,
                               capture_priority=args.capture_priority,variant=args.variant,
                               capture_source=capture_source,
                               normalize_window_scale=args.normalize_window_scale,
                               directory=directory,lead_seconds=args.lead_ms/1000,
                               lead_uncertainty=args.lead_uncertainty_ms/1000,
                               physical=True,learn_lead=args.learn_lead is not False,
                               video_recording=args.video_recording,
                               recording=not args.no_recording,**session_options)
                except KeyboardInterrupt:
                    manifest_path=directory/'manifest.json'
                    print('\nStopped.',flush=True)
                    if manifest_path.is_file():
                        manifest=json.loads(manifest_path.read_text())
                        print(json.dumps(dict(recording=str(directory),complete=manifest.get('complete'),
                            frames_written=manifest.get('frames_written'),
                            frames_dropped=manifest.get('frames_dropped'),
                            capture_sequences_skipped=manifest.get('capture_sequences_skipped'),
                            capture_source=manifest.get('capture_source'),
                            capture_source_size=manifest.get('capture_source_size'),
                            capture_frame_size=manifest.get('capture_frame_size'),
                            capture_roi=manifest.get('capture_roi'),
                            measurement_reasons=manifest.get('measurement_reasons'),
                            detector_ui_scale=manifest.get('detector_ui_scale'),
                            normalize_window_scale=manifest.get('normalize_window_scale'),
                            performance=manifest.get('performance')),indent=2))
                    return
            else:
                result=run_session(seconds=args.seconds,synthetic=args.synthetic,fps=args.fps,
                               capture_priority=args.capture_priority,variant=args.variant,
                               capture_source=capture_source,
                               normalize_window_scale=args.normalize_window_scale,
                               directory=directory,lead_seconds=args.lead_ms/1000,
                               lead_uncertainty=args.lead_uncertainty_ms/1000,
                               video_recording=args.video_recording,
                               physical=False,learn_lead=False,
                               recording=not args.no_recording,**session_options)
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','events')},indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2)+'\n')
        return
    rows = []
    if not windows_capture:
        from vd.capture import PortalCapture
        capture_factory=partial(PortalCapture,synthetic=args.synthetic,fps=args.fps,
                                priority=args.capture_priority,source=capture_source,
                                normalize_window_scale=args.normalize_window_scale)
    with capture_factory() as capture:
        # Screen selection can take up to a minute; no input device is opened.
        frame = capture.next(timeout=65)
        if frame is None:
            raise RuntimeError('No first frame')
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            rows.append(dict(sequence=frame.sequence, media_time=frame.media_time,
                received_time=frame.received_time, consumed_time=frame.consumed_time,
                copy_completed_time=frame.copy_completed_time,
                timestamp_kind=frame.timestamp_kind, pts_ns=frame.pts_ns,
                negotiated_caps=frame.negotiated_caps,
                capture_source_size=frame.source_size,
                capture_frame_size=(frame.image.shape[1], frame.image.shape[0])))
            frame = capture.next(frame.sequence, timeout=2)
            if frame is None:
                raise RuntimeError('Capture stalled')
    gaps = [(b['received_time']-a['received_time'])*1000 for a,b in zip(rows,rows[1:])]
    ages = [(r['received_time']-r['media_time'])*1000 for r in rows if r['media_time'] is not None]
    copy_ages = [(r['copy_completed_time']-r['media_time'])*1000 for r in rows
                 if r['copy_completed_time'] is not None and r['media_time'] is not None]
    result = dict(synthetic=args.synthetic, frames=len(rows),
        requested_fps=args.fps,
        received_fps=round(1000/statistics.mean(gaps),2) if gaps and sum(gaps)>0 else None,
        capture_source=capture_source,
        capture_source_size=rows[-1]['capture_source_size'] if rows else None,
        capture_frame_size=rows[-1]['capture_frame_size'] if rows else None,
        capture_roi=list(capture.roi),
        capture_physical_roi=getattr(capture,'physical_roi',None),
        negotiated_caps=sorted({r['negotiated_caps'] for r in rows}),
        delivery_gap_median_ms=statistics.median(gaps) if gaps else None,
        delivery_gap_max_ms=max(gaps) if gaps else None,
        mapped_media_age_median_ms=statistics.median(ages) if ages else None,
        mapped_media_age_p95_ms=statistics.quantiles(ages,n=20,method='inclusive')[18]
                                if len(ages)>1 else (ages[0] if ages else None),
        mapped_media_age_max_ms=max(ages) if ages else None,
        copy_age_median_ms=statistics.median(copy_ages) if copy_ages else None,
        game_render_age_measured=False,
        worker_exit_code=capture.proc.returncode,
        reader_stopped=not capture.reader.is_alive(),
        rows=rows)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
    except (RuntimeError,ValueError,OSError) as exc:
        print('VD: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
