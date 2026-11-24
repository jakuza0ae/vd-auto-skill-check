#!/usr/bin/env python3
"""Summarize solver-observed checks; does not infer game outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from vd.calibration import FreezeObserver
from vd.motion import Motion
from vd.vision import Arc,Measurement,Needle


def _events(directory):
    # Native present samples dominate long idle sessions; keep check events only.
    with (directory/'events.jsonl').open() as stream:
        for line in stream:
            if 'NATIVE_PRESENT_TIMING' not in line:
                yield json.loads(line)


def rescore_landings(directory, events):
    keys={e['generation']:e for e in events if e.get('kind')=='KEYDOWN'}
    endings={e['generation']:e for e in events if e.get('kind')=='END'}
    original={e['generation']:e['landing']['label'] for e in events
              if e.get('kind')=='CV_LANDING'}
    observers={}
    for generation,key in keys.items():
        if key.get('prefire_motion') is None:
            continue
        observers[generation]=FreezeObserver(motion=Motion(**key['prefire_motion']),
            press_at=key['at'],frame_at=key['frame_at'],great=Arc(**key['great']),
            good=Arc(**key['good']) if key.get('good') else None,
            chain=key.get('chain',1),late_dispatch=bool(key.get('outside_target_window')))
    frames=Path(directory)/'frames.jsonl'
    if frames.exists():
        with frames.open() as stream:
            for line in stream:
                row=json.loads(line)
                generation=row.get('state',{}).get('generation')
                observer=observers.get(generation)
                m=row.get('state',{}).get('measurement')
                if observer is None or not m or row['decision_time']<keys[generation]['at']:
                    continue
                if not -.002<=row['decision_time']-m['timestamp']<=.080:
                    observer.tainted='STALE_FRAME'
                observer.feed(Measurement(m['timestamp'],tuple(m['center']) if m['center'] else None,
                    m['prompt_score'],Arc(**m['great']) if m['great'] else None,
                    Arc(**m['good']) if m['good'] else None,
                    tuple(Needle(**n) for n in m['candidates']),m['reason']))
    results=[]
    for generation,observer in observers.items():
        ending=endings.get(generation)
        if ending is None:
            continue
        landing=observer.finish(ending['reason'])
        results.append(dict(generation=generation,original=original.get(generation),
            rescored=landing.label,angle=landing.angle,reason=landing.reason,
            game_outcome_measured=False))
    return dict(labels=dict(Counter(r['rescored'] for r in results)),
        changed=[r for r in results if r['original']!=r['rescored']],
        non_great=[r for r in results if r['rescored'] in ('CV_MISS','CV_GOOD')],
        game_outcome_measured=False)


def audit(directory, *, rescore=False):
    directory=Path(directory)
    manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    events=list(_events(directory))
    begins=[e for e in events if e.get('kind')=='BEGIN']
    keys=[e for e in events if e.get('kind')=='KEYDOWN']
    endings=[e for e in events if e.get('kind')=='END']
    landings=[e for e in events if e.get('kind')=='CV_LANDING']
    pressed=Counter(e['generation'] for e in keys)
    result=dict(recording=str(directory),manifest_present=manifest_path.is_file(),
        recording_complete=manifest.get('complete'),recording_state=manifest.get('recording_state'),
        detected_checks=len(begins),
        checks_with_keydown=sum(pressed[b['generation']]>0 for b in begins),
        keydowns=len(keys),duplicate_keydowns=sum(n-1 for n in pressed.values() if n>1),
        unpressed_generations=[b['generation'] for b in begins if not pressed[b['generation']]],
        end_reasons=dict(Counter(e.get('reason') for e in endings)),
        targets=dict(Counter(e.get('plan',{}).get('target_grade') for e in keys)),
        timing_modes=dict(Counter(e.get('plan',{}).get('timing_mode','PREDICTED')
                                  for e in keys)),
        cv_landings=dict(Counter(e.get('landing',{}).get('label') for e in landings)),
        game_outcomes_measured=False)
    if rescore:
        result['rescored_cv']=rescore_landings(directory,events)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings',nargs='+',type=Path)
    parser.add_argument('--since',help='Include recording directory names at or after this prefix')
    parser.add_argument('--rescore-landings',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    directories=[]
    for path in args.recordings:
        directories.extend([path] if (path/'events.jsonl').is_file() else
            sorted(p for p in path.iterdir() if (p/'events.jsonl').is_file()))
    directories=sorted(set(directories))
    result=[audit(p,rescore=args.rescore_landings) for p in directories
            if args.since is None or p.name>=args.since]
    payload=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(payload)
        print(f'Audited {len(result)} recordings; saved {args.output}')
    else:
        print(payload,end='')
