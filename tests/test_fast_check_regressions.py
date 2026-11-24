import json
import math
from pathlib import Path

import pytest

from vd.motion import MotionTracker
from vd.vision import Needle


TRACES=json.loads((Path(__file__).parent/'fixtures/fast_checks_20261015.json').read_text())


def needles(phase):
    return (Needle(phase%360,80,.9,30,2),)


@pytest.mark.parametrize(('generation','low','high'),[(31,500,600),(32,630,750),(33,850,1000),(38,420,650)])
def test_recorded_fast_motion_survives_cadence_jitter_until_original_keydown(generation,low,high):
    trace=next(entry for entry in TRACES['traces'] if entry['generation']==generation)
    tracker=MotionTracker()
    prefire=None
    for row in trace['rows']:
        measured=row['measurement']
        result=tracker.update(row['media_time'],tuple(Needle(**n) for n in measured['candidates']))
        if row['decision_time']<=trace['keydown']['at']:
            prefire=result
    assert prefire is not None
    assert low<prefire.speed<high
    assert prefire.samples>=6


@pytest.mark.parametrize('speed',[700,900,1200,1500,1800])
def test_a_frame_of_phase_jitter_keeps_fit_and_exposes_uncertainty(speed):
    tracker=MotionTracker()
    results=[]
    for index in range(30):
        at=1+index/60
        # Rendering and presentation can quantize phase at different instants.
        phase=20+speed*index/60+speed*.0035*math.sin(index*1.7)
        result=tracker.update(at,needles(phase))
        if index>=10:
            results.append(result)
    assert all(result is not None for result in results)
    assert all(abs(result.speed-speed)<speed*.08 for result in results)
    assert any(abs(result.latest_innovation)>2 for result in results)


def test_true_jump_is_still_rejected_after_jitter_tolerance():
    tracker=MotionTracker()
    for index in range(10):
        tracker.update(1+index/60,needles(20+900*index/60))
    assert tracker.estimate is not None
    assert tracker.update(1+10/60,needles(20+900*10/60+60)) is None
    assert tracker.reason=='PHASE_OUTLIER'
    assert tracker.estimate is None


def test_smaller_response_compensation_moves_recorded_fast_miss_into_white():
    from vd.motion import Motion
    from vd.planning import Planner,DEFAULT_RESPONSE_DELAY_SECONDS
    from vd.vision import Arc
    trace=next(entry for entry in TRACES['traces'] if entry['generation']==32)
    key=trace['keydown']
    motion=Motion(**key['prefire_motion'])
    target=Arc(**key['great'])
    planner=Planner(lead_seconds=DEFAULT_RESPONSE_DELAY_SECONDS,lead_uncertainty=.020)
    planner.begin(target,motion.phase)
    plan=planner.update(motion,frame_at=motion.at,now=motion.at)
    actual_cv_plateau=125.56913447380066
    observed_response=.0008591915796103746
    modelled_impact=motion.phase_at(plan.press_at+observed_response)%360
    assert not target.contains(actual_cv_plateau)
    assert target.contains(modelled_impact)
    assert .15<(modelled_impact-target.start)/target.width<.85
    assert plan.target_grade=='GREAT'
    assert planner.lead==.004
