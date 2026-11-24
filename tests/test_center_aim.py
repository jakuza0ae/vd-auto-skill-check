import importlib.util
from pathlib import Path

import pytest

from vd.calibration import FreezeObserver
from vd.engine import Engine
from vd.motion import Motion
from vd.planning import DEFAULT_RESPONSE_DELAY_SECONDS,Planner
from vd.vision import Arc,Measurement,Needle


@pytest.mark.parametrize('speed',[280,700,1000,1500,1800])
def test_white_center_with_margin_across_supported_speeds(speed):
    p=Planner(lead_seconds=DEFAULT_RESPONSE_DELAY_SECONDS,lead_uncertainty=.020)
    p.begin(Arc(120,10),70,Arc(131,42))
    m=Motion(1,80,speed,0,0,1,8)
    plan=p.update(m,frame_at=1,now=1)
    assert plan.target_grade=='GREAT'
    assert plan.target_phase==125
    assert plan.target_window_start==121
    assert plan.target_window_width==8
    assert m.phase_at(plan.press_at+DEFAULT_RESPONSE_DELAY_SECONDS)==pytest.approx(125)
    assert m.phase_at(plan.latest_press_at+DEFAULT_RESPONSE_DELAY_SECONDS)==pytest.approx(129)


def test_recorded_early_freeze_delay_setting_moves_modelled_landing_into_white():
    # Source: 20261013_225849_128579, generation 1. CV evidence only.
    m=Motion(262614.051380712,505.8478260934353,276.04778676868085,
        .4086978986534291,13.926547130706261,262614.051380712,20,-2.199497147239242)
    great=Arc(159.73821009168043,10.594063362231793)
    p=Planner(lead_seconds=DEFAULT_RESPONSE_DELAY_SECONDS,lead_uncertainty=.020)
    p.begin(great,500)
    plan=p.update(m,frame_at=262614.069062919,now=262614.0789)
    observed_response=.008187331272840393
    modelled_landing=m.phase_at(plan.press_at+observed_response)%360
    fraction=(modelled_landing-great.start)/great.width
    assert .35<fraction<.65
    assert plan.target_grade=='GREAT'
    assert plan.press_at>262614.089436054


def test_dispatch_frame_is_not_independent_freeze_observation():
    m=Motion(1,20,280,.2,1,1,9)
    observer=FreezeObserver(motion=m,press_at=1,frame_at=1,great=Arc(35,12))
    observer.feed(Measurement(1,(160,160),.99,Arc(35,12),None,
        (Needle(20,100,1,50,3),),'OK'))
    assert observer.tainted is None
    assert not observer.plateau
    observer.feed(Measurement(.99,None,0,None,None,(),'NO_PROMPT'))
    assert observer.tainted=='NONINCREASING_TIME'


@pytest.mark.parametrize('override',[None,35])
def test_vulkan_cli_uses_updated_response_default_and_preserves_override(monkeypatch,tmp_path,override):
    path=Path(__file__).resolve().parents[1]/'run-vulkan.py'
    spec=importlib.util.spec_from_file_location('vulkan_center_cli',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls=[]
    monkeypatch.setattr(module.runtime,'run_session',lambda **kwargs:calls.append(kwargs) or {})
    arguments=['--observe','--seconds','.01','--recording',str(tmp_path/'unused')]
    if override is not None:
        arguments+=['--response-delay-ms',str(override)]
    module.main(arguments)
    assert calls[0]['lead_seconds']==pytest.approx(.004 if override is None else .035)
    assert Engine().planner.lead==DEFAULT_RESPONSE_DELAY_SECONDS
