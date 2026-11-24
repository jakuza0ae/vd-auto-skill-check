import pytest

from vd.motion import Motion
from vd.planning import Planner
from vd.vision import Arc


@pytest.mark.parametrize('phase', [40,60,72])
def test_trailing_good_triggers_immediate_great_attempt_without_waiting_for_good(phase):
    p=Planner(lead_seconds=0,lead_uncertainty=.020)
    p.begin(Arc(20,10),phase,Arc(31,42))
    occurrence=p.occurrence
    assert occurrence.great_start==20
    plan=p.update(Motion(1,phase,280,0,0,1,6),frame_at=1,now=1)
    assert plan.target_grade=='GREAT'
    assert plan.target_phase==25
    assert plan.press_at==1
    assert plan.timing_mode=='LAST_CHANCE'
    assert p.occurrence==occurrence


@pytest.mark.parametrize('speed',[280,700,1000,1500,1800])
def test_uncertain_fast_fit_still_aims_white_center(speed):
    p=Planner(lead_seconds=.005,lead_uncertainty=.020)
    p.begin(Arc(120,10),60,Arc(131,42))
    plan=p.update(Motion(1,70,speed,1.5,200,1,2),frame_at=1,now=1)
    assert plan is not None
    assert plan.target_grade=='GREAT'
    assert plan.target_phase==125
    assert plan.uncertainty_degrees>5
    assert 120<=70+speed*(plan.intended_press_at+.005-1)<=130


def test_crossed_center_attempts_immediately_instead_of_waiting_near_exit():
    p=Planner(lead_seconds=0,lead_uncertainty=.020)
    p.begin(Arc(20,10),300,Arc(31,42))
    plan=p.update(Motion(1,386,280,2,20,1,6),frame_at=1,now=1)
    assert plan.target_grade=='GREAT'
    assert plan.target_window_start==380
    assert plan.target_phase==385
    assert plan.press_at==1
    assert plan.timing_mode=='LAST_CHANCE'


def test_latched_occurrence_never_rearms_after_white_passes():
    p=Planner(lead_seconds=0,lead_uncertainty=.003)
    p.begin(Arc(20,10),300,Arc(31,42))
    occurrence=p.occurrence
    assert p.update(Motion(1,360,280,0,2,1,6),frame_at=1,now=1)
    for at,phase in [(1.1,391),(2.4,750)]:
        plan=p.update(Motion(at,phase,280,0,2,at,6),frame_at=at,now=at)
        assert plan.target_grade=='GREAT'
        assert plan.timing_mode=='LAST_CHANCE'
        assert plan.press_at==at
        assert p.occurrence==occurrence


def test_chained_passed_white_tries_immediately_instead_of_aiming_good():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(20,10),32,Arc(31,42),occurrence_start=20)
    plan=p.update(Motion(1,32,700,0,0,1,6),frame_at=1,now=1)
    assert plan.target_grade=='GREAT'
    assert plan.press_at==1
    assert plan.timing_mode=='LAST_CHANCE'
    assert p.occurrence.great_start==20


def test_dispatch_after_white_deadline_is_rejected_even_with_fresh_frame():
    p=Planner(lead_seconds=0,lead_uncertainty=0)
    p.begin(Arc(100,10),80,Arc(111,42))
    plan=p.update(Motion(1,80,1000,0,0,1,6),frame_at=1,now=1)
    assert plan.press_at==pytest.approx(1.025)
    assert plan.latest_press_at==pytest.approx(1.029)
    assert not p.claim(plan,now=1.040,held=True,capture_alive=True)
    assert not p.fired



def test_slightly_crossed_center_presses_without_further_tail_delay():
    p=Planner(lead_seconds=0,lead_uncertainty=0)
    p.begin(Arc(120,10),100)
    plan=p.update(Motion(1,125.1,1000,0,0,1,6),frame_at=1,now=1)
    assert plan.target_phase==125
    assert plan.press_at==1
    assert plan.timing_mode=='LAST_CHANCE'


@pytest.mark.parametrize('innovation',[float('nan'),float('inf')])
def test_invalid_latest_phase_confidence_cancels_timer(innovation):
    p=Planner(lead_seconds=0,lead_uncertainty=0)
    p.begin(Arc(120,10),100)
    old=p.update(Motion(1,100,1000,0,0,1,6),frame_at=1,now=1)
    assert old is not None
    assert p.update(Motion(1.01,110,1000,0,0,1.01,6,innovation),frame_at=1.01,now=1.01) is None
    assert p.current is None
    assert p.reason=='INVALID_TIMING'


@pytest.mark.parametrize('speed',[280,700,1000,1500,1800])
@pytest.mark.parametrize('phase',[131,200])
def test_late_attempt_never_waits_for_good_at_any_supported_speed(speed,phase):
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(120,10),100,Arc(131,42))
    plan=p.update(Motion(1,phase,speed,0,0,1,6),frame_at=1,now=1)
    assert plan.target_grade=='GREAT'
    assert plan.target_phase==125
    assert plan.target_window_start==120
    assert plan.target_window_width==10
    assert plan.press_at==1
    assert plan.timing_mode=='LAST_CHANCE'
    assert plan.fallback_reason=='GREAT_PASSED'
    assert p.claim(plan,now=1,held=True,capture_alive=True)
    assert not p.claim(plan,now=1.001,held=True,capture_alive=True)
