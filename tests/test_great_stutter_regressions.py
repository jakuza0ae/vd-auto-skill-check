import pytest

from vd.engine import Engine
from vd.motion import MotionTracker
from vd.vision import Arc,Measurement,Needle


def needles(angle):
    return (Needle(angle%360,80,.9,30,2),)


def observation(at,angle,target=130):
    return Measurement(at,(160,162.5),.99,Arc(target,10),Arc(target+11,42),needles(angle),'OK')


def test_stall_replaces_old_prediction_with_one_best_effort_attempt():
    e=Engine(lead_seconds=0,lead_uncertainty=.003)
    for i,phase in enumerate([30,48,66,84,102]):
        at=1+i*.020
        e.observe(observation(at,phase),now=at)
    old=e.planner.current
    assert old is not None
    e.observe(observation(1.097,102),now=1.097)
    e.observe(observation(1.110,102),now=1.110)
    assert e.planner.current is None
    attempt=e.poll(old.press_at)
    assert attempt is not old and attempt.timing_mode=='LAST_CHANCE'
    assert e.poll(old.press_at+.001) is None


def test_normal_duplicate_survives_but_confirmed_stall_cancels():
    e=Engine(lead_seconds=0,lead_uncertainty=.003)
    for i,phase in enumerate([30,36,42,48,54,60,66,72,78,84,90]):
        at=1+i*.020
        e.observe(observation(at,phase,96),now=at)
    old=e.planner.current
    assert old is not None
    e.observe(observation(1.217,90,96),now=1.217)
    assert e.motion.reason=='REPEATED_ANGLE'
    assert e.planner.current is not None
    e.observe(observation(1.234,90,96),now=1.234)
    assert e.motion.reason=='STALLED'
    assert e.planner.current is None
    assert e.poll(old.press_at).timing_mode=='LAST_CHANCE'
    e.observe(observation(1.400,90,96),now=1.400)
    e.observe(observation(1.700,90,96),now=1.700)
    assert e.poll(1.700) is None


def test_60hz_capture_30hz_animation_duplicates_are_not_outliers():
    t=MotionTracker()
    for i in range(12):
        t.update(1+i/60,needles((i//2)*1000/30))
    assert t.estimate is not None
    assert t.estimate.speed==pytest.approx(1000,rel=.01)
    assert t.reason=='REPEATED_ANGLE'
    last=t.last_observed_motion_at
    phase=t.last_angle
    assert t.update(last+1/30,needles(phase)) is not None
    assert t.update(last+.050,needles(phase)) is None
    assert t.reason=='STALLED'


def test_resume_requires_two_independent_moving_samples_and_keeps_turn():
    t=MotionTracker()
    for i,angle in enumerate([340,350,0,10,20]):
        t.update(1+i*.02,needles(angle))
    floor=t.last_unwrapped
    assert t.update(1.120,needles(20)) is None
    assert t.reason=='STALLED'
    assert t.update(1.140,needles(20)) is None
    assert not t.points
    assert t.update(1.160,needles(30)) is None
    assert len(t.points)==1
    motion=t.update(1.180,needles(40))
    assert motion is not None and motion.samples==2
    assert motion.phase>floor>360


def test_valid_fast_105_degree_advance_does_not_reset_fit():
    t=MotionTracker()
    for i,phase in enumerate([0,30,60,90]):
        t.update(1+i*.02,needles(phase))
    motion=t.update(1.130,needles(195))
    assert motion is not None
    assert motion.speed==pytest.approx(1500)
    assert motion.phase==pytest.approx(195)


def test_latest_acceleration_phase_is_preserved_in_confidence():
    t=MotionTracker()
    for i in range(10):
        t.update(1+i*.02,needles(30+900*i*.02))
    motion=t.update(1.20,needles(214))
    assert motion is not None
    assert motion.phase==214
    assert motion.latest_innovation==pytest.approx(4)
    e=Engine(lead_seconds=0,lead_uncertainty=0)
    e.planner.begin(Arc(225,10),192,Arc(236,42))
    plan=e.planner.update(motion,frame_at=1.20,now=1.20)
    actual=214+1100*(plan.press_at-1.20)
    assert 225<=actual<=235
    assert plan.uncertainty_degrees>=4


def test_missing_needle_discards_fit_until_two_fresh_movements():
    t=MotionTracker()
    for i,angle in enumerate([10,20,30,40,50]):
        t.update(1+i*.02,needles(angle))
    assert t.estimate is not None
    assert t.update(1.10,()) is None
    assert not t.points
    assert t.update(1.12,needles(60)) is None
    assert t.update(1.14,needles(70)) is not None


@pytest.mark.parametrize('speed',[280,500,1000,1500,1800])
@pytest.mark.parametrize('origin',[1.,180000.])
def test_supported_motion_speeds_keep_latest_phase(speed,origin):
    t=MotionTracker()
    for index in range(12):
        at=origin+index*.02
        phase=20+speed*index*.02
        result=t.update(at,needles(phase))
    assert result is not None
    assert result.speed==pytest.approx(speed)
    assert result.phase==pytest.approx(phase)



def test_speed_tolerance_does_not_accept_real_limit_exceedance():
    t=MotionTracker()
    for index in range(12):
        result=t.update(1+index*.02,needles(20+1800.01*index*.02))
    assert result is None
    assert t.estimate is None


@pytest.mark.parametrize('invalid',[ 'stale','reordered'])
def test_invalid_frame_time_cancels_fit_and_requires_two_new_movements(invalid):
    e=Engine(lead_seconds=0,lead_uncertainty=0)
    for index in range(7):
        at=1+index*.02
        e.observe(observation(at,30+index*6),now=at)
    assert e.motion.estimate is not None
    old=e.planner.current
    assert old is not None
    if invalid=='stale':
        e.observe(observation(1.14,72),now=1.23)
        assert e.reason=='STALE_FRAME'
    else:
        e.observe(observation(1.12,66),now=1.12)
        assert e.reason=='NONINCREASING_TIME'
    assert e.motion.estimate is None and e.planner.current is None
    assert e.poll(old.press_at) is None
    e.observe(observation(1.24,78),now=1.24)
    assert e.planner.current is None
    e.observe(observation(1.26,84),now=1.26)
    assert e.motion.estimate is not None
    assert e.planner.current is not None
