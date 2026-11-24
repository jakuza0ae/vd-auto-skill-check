import json

import pytest
from pathlib import Path
import threading
from types import SimpleNamespace

from vd.dispatch import dispatch
from vd.engine import Engine
from vd.motion import Motion
from vd.vision import Arc,Measurement,Needle


def observation(at,phase=None,visible=True):
    return Measurement(at,(160,162.5) if visible else None,.99 if visible else 0,
        Arc(130,10) if visible else None,Arc(141,42) if visible else None,
        () if phase is None else (Needle(phase,80,.9,30,2),),'OK')


def test_recorded_missed_check_gets_one_attempt_even_with_new_native_frames_pending():
    trace=json.loads((Path(__file__).parent/'fixtures/missed_check_20261013.json').read_text())
    engine=Engine(lead_seconds=trace['initial_response_delay'],lead_uncertainty=.023)
    capture=SimpleNamespace(cv=threading.Condition(),stopping=False,error=None,
        proc=SimpleNamespace(poll=lambda:None),has_newer=lambda _:True,
        prediction_current=lambda _:True)
    now=trace['rows'][0]['decision_time']
    output_times=[]
    class Output:
        def pulse(self):
            output_times.append(now)
            return SimpleNamespace(requested_at=now,syn_completed_at=now)
    mouse=SimpleNamespace(lock=threading.RLock(),snapshot=lambda:SimpleNamespace(healthy=True,held=True))
    output=Output()
    all_events=[]
    for row in trace['rows']:
        now=row['decision_time']
        capture.latest=SimpleNamespace(sequence=row['sequence'],media_time=row['media_time'],synthetic=False)
        plan=engine.planner.current
        if plan is not None and now>=plan.press_at:
            dispatch(engine,capture,row['sequence'],mouse=mouse,output=output,clock=lambda:now)
        data=row['measurement']
        measured=Measurement(data['timestamp'],tuple(data['center']) if data['center'] else None,
            data['prompt_score'],Arc(**data['great']) if data['great'] else None,
            Arc(**data['good']) if data['good'] else None,
            tuple(Needle(**needle) for needle in data['candidates']),data['reason'])
        engine.observe(measured,now=now,held=row['held'])
        dispatch(engine,capture,row['sequence'],mouse=mouse,output=output,clock=lambda:now)
        all_events.extend(engine.take_events())
    keys=[event for event in all_events if event['kind']=='KEYDOWN']
    assert len(keys)==len(output_times)==1
    assert keys[0]['plan']['target_grade']=='GREAT'
    assert not any(event['kind']=='MISSED_END_DIAGNOSTIC' for event in all_events)


def test_new_native_count_does_not_cancel_pending_timer(tmp_path):
    from test_vulkan_capture_source import write_snapshot
    from vd.vulkan_capture import VulkanCapture
    shm=tmp_path/'vd_layer_shm.dat'
    write_snapshot(shm,capture_count=7,capture_ts_ns=1_000_000_000,epoch=10)
    with VulkanCapture(shm_path=shm) as capture:
        frame=capture.next(timeout=.01)
        engine=Engine(lead_seconds=.1,lead_uncertainty=0)
        engine.planner.begin(Arc(96,8),60)
        plan=engine.planner.update(Motion(1,70,300,0,0,1,6),frame_at=1,now=1)
        write_snapshot(shm,capture_count=8,capture_ts_ns=1_001_000_000,epoch=10)
        assert capture.has_newer(frame.sequence)
        assert capture.prediction_current(frame)
        assert dispatch(engine,capture,frame.sequence,clock=lambda:1.) is plan
        assert engine.planner.fired


def test_first_missing_ring_frame_still_gets_one_attempt():
    engine=Engine(lead_seconds=0)
    engine.observe(observation(1,30),now=1)
    engine.observe(observation(1.02,36),now=1.02)
    engine.observe(observation(1.20,visible=False),now=1.20)
    plan=engine.poll(1.20)
    assert plan.timing_mode=='LAST_CHANCE'
    assert engine.poll(1.21) is None
    engine.observe(observation(1.22,visible=False),now=1.22)
    assert any(event['kind']=='END' and event['pressed'] for event in engine.take_events())


@pytest.mark.parametrize('during_cv',[False,True])
def test_due_timer_runs_before_replanning_the_next_frame(monkeypatch,tmp_path,during_cv):
    import numpy as np
    from vd import runtime,input as vd_input
    from vd.capture import Frame
    from test_runtime_dispatch_calibration import _Capture,_HeldMouse
    clock=SimpleNamespace(now=1.)
    monkeypatch.setattr(runtime,'time',SimpleNamespace(monotonic=lambda:clock.now,
        perf_counter=lambda:clock.now,process_time=lambda:0.,
        sleep=lambda seconds:setattr(clock,'now',clock.now+seconds)))
    class Capture(_Capture):
        def next(self,after=-1,timeout=1.):
            clock.now+=min(.005,max(.001,timeout))
            now=clock.now
            self.latest=Frame(after+1,np.zeros((240,320,3),np.uint8),
                now,now,now,'test_clock',False,None,'test_caps')
            return self.latest
    class ArmedEngine(Engine):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            now=clock.now
            self.planner.lead=0
            self.target=Arc(96,8)
            self.center=(160,162.5)
            self.active=True
            self.planner.begin(self.target,70)
            self.planner.update(Motion(now,97 if during_cv else 100,300,0,0,now,6),frame_at=now,now=now)
        def observe(self,*args,**kwargs):
            if not during_cv:
                assert output.pulses==1
            return super().observe(*args,**kwargs)
    class Output:
        error=None
        pulses=0
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def pulse(self):
            self.pulses+=1
            now=clock.now
            return SimpleNamespace(requested_at=now,syn_completed_at=now)
    output=Output()
    class Detector:
        roi_offset=(0,0)
        def __init__(self,**_): self.first=True
        def measure(self,image,timestamp,**_):
            if during_cv and self.first:
                assert output.pulses==0
                clock.now+=.012
            else:
                assert output.pulses==1, 'Pending Space must precede expensive CV'
            self.first=False
            return Measurement(timestamp,None,0,None,None,(),'NO_PROMPT')
    monkeypatch.setattr(runtime,'Engine',ArmedEngine)
    monkeypatch.setattr(runtime,'PortalCapture',Capture)
    monkeypatch.setattr(runtime,'Detector',Detector)
    monkeypatch.setattr(vd_input,'MouseMonitor',_HeldMouse)
    monkeypatch.setattr(vd_input,'SpaceOutput',lambda:output)
    result=runtime.run_session(seconds=.025,synthetic=False,fps=60,directory=tmp_path/'priority',
        lead_seconds=0,lead_uncertainty=0,physical=True,recording=False,video_recording=False)
    assert result['physical_presses']==output.pulses==1


def test_overdue_great_timer_still_presses_immediately_without_aiming_good():
    e=Engine(lead_seconds=0,lead_uncertainty=0)
    e.target=Arc(100,10)
    e.good=Arc(111,42)
    e.center=(160,162.5)
    e.active=True
    e.last_timestamp=1
    motion=Motion(1,80,1000,0,0,1,6)
    e.motion.estimate=motion
    e.planner.begin(e.target,80,e.good)
    old=e.planner.update(motion,frame_at=1,now=1)
    assert old.latest_press_at<1.04
    attempt=e.poll(1.04)
    assert attempt.target_grade=='GREAT'
    assert attempt.target_phase==105
    assert attempt.timing_mode=='LAST_CHANCE'
    assert attempt.press_at==1.04
    assert e.poll(1.041) is None


def test_completed_cv_corrects_timer_before_dispatch_and_keeps_active_cost(monkeypatch,tmp_path):
    import numpy as np
    from vd import runtime
    from vd.capture import Frame
    clock=SimpleNamespace(now=1.)
    monkeypatch.setattr(runtime,'time',SimpleNamespace(
        monotonic=lambda:clock.now,perf_counter=lambda:clock.now,process_time=lambda:0.))
    observed=[]
    dispatched=[]
    class ArmedEngine(Engine):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.active=True
            self.center=(160,162.5)
            self.target=Arc(96,8)
            self.planner.begin(self.target,97)
            self.planner.update(Motion(1,97,300,0,0,1,6),frame_at=1,now=1)
        def observe(self,m,**kwargs):
            observed.append(m)
            # The completed observation puts the needle behind the old forecast.
            self.last_timestamp=m.timestamp
            self.planner.update(Motion(m.timestamp,90,300,0,0,m.timestamp,6),
                                frame_at=m.timestamp,now=clock.now)
    class Capture:
        roi=None
        def __init__(self,**_): self.first=True
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def next(self,after=-1,timeout=1.):
            if self.first:
                self.first=False
                return Frame(0,np.zeros((240,320,3),np.uint8),1,1,1,
                             'test_clock',False,None,'test_caps')
            clock.now+=max(.001,timeout)
            return None
    class Detector:
        roi_offset=(0,0)
        def __init__(self,**_): pass
        def measure(self,image,timestamp,**_):
            clock.now+=.012
            return observation(timestamp,90)
    def dispatch_current(engine,*args,**kwargs):
        assert observed, 'Completed CV must be applied before consulting the old timer'
        assert engine.planner.current.press_at>clock.now
        dispatched.append(engine.planner.current)
    monkeypatch.setattr(runtime,'Engine',ArmedEngine)
    monkeypatch.setattr(runtime,'PortalCapture',Capture)
    monkeypatch.setattr(runtime,'Detector',Detector)
    monkeypatch.setattr(runtime,'dispatch',dispatch_current)
    result=runtime.run_session(seconds=.020,synthetic=False,fps=60,
        directory=tmp_path/'fresh-cv',lead_seconds=0,lead_uncertainty=0,
        recording=False,video_recording=False)
    assert len(observed)==len(dispatched)==1
    assert dispatched[0].target_grade=='GREAT'
    assert result['performance']['active_cv_ms']['p50']==12.
    assert result['performance']['active_source_age_ms']['p50']==12.
