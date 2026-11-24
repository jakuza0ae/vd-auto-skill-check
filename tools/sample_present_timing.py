#!/usr/bin/env python3
"""Record native present counters without activating a capture consumer."""
import argparse
import json
from pathlib import Path
import time


def sample(status_path, output, seconds):
    end=time.monotonic()+seconds
    previous=None
    with output.open('x') as target:
        while time.monotonic()<end:
            try:
                with status_path.open() as source:
                    status=json.loads(source.read(65536))
                stamp=status.get('status_time_ns')
                now=time.monotonic()
                if (isinstance(stamp,int) and stamp!=previous
                        and 0<=now-stamp/1e9<=2 and status.get('present_timing')):
                    target.write(json.dumps(dict(kind='NATIVE_PRESENT_TIMING',at=now,status=status))+'\n')
                    target.flush()
                    previous=stamp
            except (OSError,ValueError):
                pass
            time.sleep(min(.250,max(0,end-time.monotonic())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--status',type=Path,default=Path.home()/'.var/app/org.vinegarhq.Sober/data/vulkan/vd_layer_status.json')
    args=parser.parse_args()
    if not 0<args.seconds<=3600:
        parser.error('--seconds must be in (0, 3600]')
    sample(args.status,args.output,args.seconds)
