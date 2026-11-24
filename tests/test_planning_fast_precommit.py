import pytest

from vd.motion import Motion
from vd.planning import Planner
from vd.vision import Arc


def test_subframe_great_updates_confidence_without_moving_consistent_deadline():
    p = Planner(lead_seconds=0, lead_uncertainty=.003)
    p.begin(Arc(20, 10), 300, Arc(31, 42))

    committed = p.update(
        Motion(1.000, 350.0, 900, .10, 4, 1.000, 10),
        frame_at=1.000,
        now=1.000,
    )
    noisy = p.update(
        Motion(1.017, 365.3, 900, .90, 35, 1.017, 11),
        frame_at=1.017,
        now=1.017,
    )

    assert committed is not None
    assert committed.target_grade == 'GREAT'
    assert committed.press_at == pytest.approx(1.038888888888889)
    assert noisy is not committed
    assert noisy.press_at==pytest.approx(committed.press_at)
    assert noisy.uncertainty_degrees>committed.uncertainty_degrees
    assert noisy.target_grade == 'GREAT'


def test_subframe_great_does_not_keep_timer_when_fresh_mean_misses_window():
    p = Planner(lead_seconds=0, lead_uncertainty=.003)
    p.begin(Arc(20, 10), 300, Arc(31, 42))

    committed = p.update(
        Motion(1.000, 350.0, 900, .10, 4, 1.000, 10),
        frame_at=1.000,
        now=1.000,
    )
    diverted = p.update(
        Motion(1.017, 367.5, 1050, .90, 35, 1.017, 11),
        frame_at=1.017,
        now=1.017,
    )

    assert committed is not None and committed.target_grade == 'GREAT'
    assert diverted is not committed
    assert diverted is not None and diverted.target_grade=='GREAT'
    assert diverted.press_at<committed.press_at
    assert 380<=367.5+1050*(diverted.press_at-1.017)<=390



def test_fresh_phase_retimes_center_even_when_old_timer_would_still_hit_white():
    p=Planner(lead_seconds=0,lead_uncertainty=0)
    p.begin(Arc(20,10),300)
    old=p.update(Motion(1,350,900,.1,4,1,10),frame_at=1,now=1)
    fresh=Motion(1.017,367,900,.1,4,1.017,11,latest_innovation=1.7)
    assert 380<fresh.phase_at(old.press_at)<390
    new=p.update(fresh,frame_at=1.017,now=1.017)
    assert new.press_at<old.press_at
    assert fresh.phase_at(new.press_at)==pytest.approx(385)
    assert new.uncertainty_degrees>=1.7
