#!/usr/bin/env python3
"""Compare recorded evidence; CV labels never count as confirmed game verdicts."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


VERDICTS={'GREAT','GOOD','MISS'}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    if not values:
        return {'samples':0}
    values=sorted(values)
    return dict(samples=len(values),mean_ms=sum(values)/len(values),
        **{f'p{p}_ms':values[max(0,math.ceil(p/100*len(values))-1)] for p in (50,95,99)},
        max_ms=values[-1])


def present_intervals(events):
    """Difference cumulative native counters; discard pre-recording history."""
    previous={}
    histogram=[0]*256
    total_ns=0
    segments=0
    for event in events:
        if event.get('kind')!='NATIVE_PRESENT_TIMING':
            continue
        status=event['status']
        timing=status['present_timing']
        key=(status['pid'],status['init_timestamp_ns'],timing['generation'])
        old=previous.get(key)
        previous[key]=timing
        counts=timing['histogram']
        if len(counts)!=256 or timing['bucket_width_ns']!=1_000_000:
            raise ValueError('Unsupported present timing histogram')
        if old is None:
            continue
        delta=[b-a for a,b in zip(old['histogram'],counts)]
        samples=timing['samples']-old['samples']
        elapsed=timing['sum_ns']-old['sum_ns']
        if any(n<0 for n in delta) or samples<0 or elapsed<0 or sum(delta)!=samples:
            raise ValueError('Non-monotonic native timing counters')
        if not samples:
            continue
        histogram=[a+b for a,b in zip(histogram,delta)]
        total_ns+=elapsed
        segments+=1
    count=sum(histogram)
    result=dict(samples=count,segments=segments,
        measurement='interval_between_vkQueuePresentKHR_calls',
        display_or_gpu_frame_time_measured=False,
        percentile_bucket_ms=1,first_snapshot_excluded=True)
    if count:
        result['mean_ms']=total_ns/count/1e6
        for percentile in (50,95,99):
            rank=math.ceil(percentile/100*count)
            seen=0
            for index,n in enumerate(histogram):
                seen+=n
                if seen>=rank:
                    result[f'p{percentile}_lower_ms']=index
                    result[f'p{percentile}_upper_ms']=index+1 if index<255 else None
                    break
    return result


def confirmed_outcomes(path, generations):
    if path is None:
        return dict(confirmed={},unknown=len(generations),great_rate=None)
    verdicts={}
    seen=set()
    with Path(path).open(newline='') as source:
        for row in csv.DictReader(source):
            generation=int(row['generation'])
            label=row['outcome'].strip().upper()
            if generation not in generations or generation in seen:
                raise ValueError(f'Unknown or duplicate generation {generation}')
            seen.add(generation)
            if not label or label=='UNKNOWN':
                continue
            if label not in VERDICTS or not row['evidence'].strip():
                raise ValueError('Each confirmed verdict needs GREAT/GOOD/MISS and evidence')
            verdicts[generation]=label
    counts=Counter(verdicts.values())
    return dict(confirmed=dict(counts),unknown=len(generations)-len(verdicts),
        great_rate=counts['GREAT']/len(verdicts) if verdicts else None)


def summarize(directory, outcomes=None, native_events=None):
    directory=Path(directory)
    if not (directory/'events.jsonl').is_file():
        raise FileNotFoundError(directory/'events.jsonl')
    events=read_jsonl(directory/'events.jsonl')
    frames=read_jsonl(directory/'frames.jsonl')
    manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    begins={e['generation'] for e in events if e.get('kind')=='BEGIN'}
    keys=[e for e in events if e.get('kind')=='KEYDOWN']
    generations=begins|{e['generation'] for e in keys}
    delays={name:[] for name in ('source_age','gpu_copy_completion','publication','reader_delivery')}
    for row in frames:
        pairs={'source_age':('decision_time','media_time'),
            'gpu_copy_completion':('copy_completed_time','media_time'),
            'publication':('published_time','copy_completed_time'),
            'reader_delivery':('received_time','published_time')}
        for name,(end,start) in pairs.items():
            a,b=row.get(end),row.get(start)
            if isinstance(a,(int,float)) and isinstance(b,(int,float)) and a>=b:
                delays[name].append((a-b)*1000)
    return dict(recording=str(directory),variant=manifest.get('variant'),
        source_sha256=manifest.get('source_sha256'),detected_checks=len(begins),
        keydowns=len(keys),intended_targets=dict(Counter(e.get('plan',{}).get('target_grade','UNKNOWN') for e in keys)),
        cv_predictions=dict(Counter(e.get('landing',{}).get('label','UNKNOWN') for e in events if e.get('kind')=='CV_LANDING')),
        game_outcomes=confirmed_outcomes(outcomes,generations),
        capture_timing={name:distribution(values) for name,values in delays.items()},
        present_intervals=present_intervals(read_jsonl(Path(native_events)) if native_events else events),
        telemetry_rows=len(frames),recording_complete=manifest.get('complete'),
        events_dropped=manifest.get('events_dropped'),telemetry_dropped=manifest.get('frames_dropped'))


def write_outcome_template(directory, output):
    events=read_jsonl(Path(directory)/'events.jsonl')
    generations=sorted({e['generation'] for e in events if e.get('kind') in ('BEGIN','KEYDOWN')})
    with Path(output).open('x',newline='') as target:
        writer=csv.writer(target)
        writer.writerow(['generation','outcome','evidence'])
        writer.writerows((generation,'UNKNOWN','') for generation in generations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path)
    parser.add_argument('after',type=Path,nargs='?')
    parser.add_argument('--write-outcome-template',type=Path)
    parser.add_argument('--outcomes-before',type=Path)
    parser.add_argument('--outcomes-after',type=Path)
    parser.add_argument('--native-before',type=Path)
    parser.add_argument('--native-after',type=Path)
    args=parser.parse_args()
    if args.write_outcome_template:
        if args.after is not None:
            parser.error('Template export takes only one recording')
        write_outcome_template(args.before,args.write_outcome_template)
        return
    if args.after is None:
        parser.error('Supply both before and after recordings')
    result=dict(before=summarize(args.before,args.outcomes_before,args.native_before),
                after=summarize(args.after,args.outcomes_after,args.native_after),
                comparable_scene_and_settings_verified=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
