from contextlib import nullcontext
from concurrent.futures import Future
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vd import runtime
from vd.calibration import FreezeObserver
from vd.capture import Frame
from vd.engine import Engine
from vd.motion import Motion
from vd.vision import Arc,Measurement,Needle


TRACES=json.loads((Path(__file__).parent/'fixtures/rare_misses_20261019.json').read_text())['traces']


@pytest.mark.parametrize('trace',TRACES,ids=lambda t:f"{t['session']}-{t['generation']}")
def test_landing_belongs_to_visible_ring_and_preserves_real_overshoots(trace):
    key=trace['keydown']
    observer=FreezeObserver(motion=Motion(**key['prefire_motion']),press_at=key['at'],
        frame_at=key['frame_at'],great=Arc(**key['great']),good=Arc(**key['good']))
    for row in trace['rows']:
        m=row['measurement']
        if not m or row['decision_time']<key['at']:
            continue
        observer.feed(Measurement(m['timestamp'],tuple(m['center']) if m['center'] else None,
            m['prompt_score'],Arc(**m['great']) if m['great'] else None,
            Arc(**m['good']) if m['good'] else None,
            tuple(Needle(**n) for n in m['candidates']),m['reason']))
    landing=observer.finish(trace['end']['reason'])
    if trace['original_landing']['label']=='CV_GOOD':
        assert landing.label=='CV_GOOD'
    elif trace['original_landing']['label']=='UNCONFIRMED':
        # Newly retained same-ring plateaus: two lagged tracks and one small overshoot.
        assert landing.label=='CV_MISS'
        assert not Arc(**key['great']).contains(landing.angle)
    elif key['prefire_motion']['speed']>850:
        assert landing.label=='CV_MISS'
        assert key['great']['start']+key['great']['width']<landing.angle
    else:
        assert landing.label=='CV_GREAT'
        assert Arc(**key['great']).contains(landing.angle)


def test_prompt_loss_without_a_plateau_does_not_invent_a_landing():
    observer=FreezeObserver(motion=Motion(1,20,900,.2,3,1,10),
        press_at=1,frame_at=1,great=Arc(35,10))
    for at in (1.02,1.04,1.06):
        observer.feed(Measurement(at,None,0,None,None,(Needle(140,80,1,40,2),),'NO_PROMPT'))
    assert observer.finish('RING_ENDED').label=='UNCONFIRMED'


@pytest.mark.parametrize(('busy_for','release','invalidate'),[(0,False,False),(.004,False,False),
    (.004,True,False),(0,False,True),(.050,False,False)])
def test_timer_precedes_cv_but_keeps_final_source_and_mouse_checks(monkeypatch,busy_for,release,invalidate):
    clock=SimpleNamespace(now=1.)
    sleeps=[]
    def monotonic():
        clock.now+=.000001
        return clock.now
    def sleep(seconds):
        assert seconds>=0
        sleeps.append(seconds)
        clock.now+=seconds
    monkeypatch.setattr(runtime,'time',SimpleNamespace(monotonic=monotonic,sleep=sleep))
    engine=Engine(lead_seconds=0,lead_uncertainty=0)
    engine.target=Arc(40,10)
    engine.center=(160,160)
    engine.active=True
    engine.last_timestamp=1
    engine.planner.begin(engine.target,42.3)
    engine.motion.estimate=Motion(1,42.3,900,.1,2,1,12)
    plan=engine.planner.update(engine.motion.estimate,frame_at=1,now=1)
    frame=SimpleNamespace(sequence=1,media_time=1.,synthetic=False)
    capture=SimpleNamespace(cv=threading.Condition(),stopping=False,error=None,
        proc=SimpleNamespace(poll=lambda:None),latest=frame,
        prediction_current=lambda _:not invalidate and clock.now>=1.003+busy_for,
        prediction_busy=lambda:clock.now<1.003+busy_for and not invalidate)
    mouse=SimpleNamespace(lock=threading.RLock(),snapshot=lambda:SimpleNamespace(
        healthy=True,held=not (release and clock.now>=1.002)))
    presses=[]
    class Output:
        def pulse(self):
            presses.append(clock.now)
            return SimpleNamespace(requested_at=clock.now,syn_completed_at=clock.now)
    assert runtime._service_timer_before_cv(engine,capture,1,mouse=mouse,
        output=Output(),cv_budget=.008)
    if release or invalidate or busy_for>.008:
        assert not presses
    else:
        assert len(presses)==1
        assert presses[0]==pytest.approx(plan.press_at+busy_for,abs=.00021)
        assert engine.planner.fired
        assert engine.take_events()[-1]['plan']['target_grade']=='GREAT'
    # Includes the fake clock ticks spent in final status checks.
    assert clock.now<=1.00901
    assert len(sleeps)<50


def test_cv_is_not_deferred_for_a_distant_target(monkeypatch):
    engine=Engine(lead_seconds=0)
    engine.planner.begin(Arc(40,10),20)
    engine.planner.update(Motion(1,20,900,.1,2,1,12),frame_at=1,now=1)
    monkeypatch.setattr(runtime,'time',SimpleNamespace(monotonic=lambda:1.,
        sleep=lambda _:pytest.fail('A distant deadline must allow fresh CV')))
    assert not runtime._service_timer_before_cv(engine,None,1,mouse=None,output=None,cv_budget=.008)


def test_dxgi_pre_cv_budget_reserves_time_for_measured_slow_frames():
    samples = [8.0, 15.83, 52.378]
    assert runtime._cv_budget_seconds(samples, windows_capture=True) == pytest.approx(.053378)
    assert runtime._cv_budget_seconds(samples, windows_capture=False) == pytest.approx(.012)


@pytest.mark.parametrize('source_state', [
    'same_source', 'changed_epoch', 'changed_generation', 'stale', 'released',
    'at_latest_press', 'after_latest_press', 'busy_then_ready',
])
def test_dxgi_timer_with_slow_cv_and_newer_capture(monkeypatch, source_state):
    clock = SimpleNamespace(now=1.0)
    anchor = SimpleNamespace(sequence=1, media_time=1.0, synthetic=False,
                             source_epoch=(41, 1), source_generation=1)
    newer = SimpleNamespace(sequence=2, media_time=1.017, synthetic=False,
                            source_epoch=(41, 1), source_generation=1)
    if source_state == 'changed_epoch':
        newer.source_epoch = (42, 1)
    elif source_state == 'changed_generation':
        newer.source_generation = 2
    elif source_state == 'stale':
        newer.media_time = .9
    capture = SimpleNamespace(cv=threading.Condition(), stopping=False, error=None,
                              proc=SimpleNamespace(poll=lambda: None), latest=anchor)
    waits = []
    capture.prediction_busy = lambda: source_state == 'busy_then_ready' and clock.now < 1.026
    capture.prediction_current = lambda frame: frame is capture.latest and not capture.prediction_busy()

    def monotonic():
        clock.now += .000001
        if clock.now >= 1.017:
            capture.latest = newer
        return clock.now

    def sleep(seconds):
        assert seconds >= 0
        waits.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(runtime, 'time', SimpleNamespace(monotonic=monotonic, sleep=sleep))
    engine = Engine(lead_seconds=0, lead_uncertainty=0)
    engine.target = Arc(40, 10)
    engine.center = (160, 160)
    engine.active = True
    engine.last_timestamp = anchor.media_time
    engine.planner.begin(engine.target, 22.5)
    engine.motion.estimate = Motion(1.0, 22.5, 900, .1, 2, 1.0, 12)
    plan = engine.planner.update(engine.motion.estimate, frame_at=1.0, now=1.0)
    assert plan.press_at == pytest.approx(1.025)
    def snapshot():
        if source_state in ('at_latest_press', 'after_latest_press'):
            clock.now = plan.latest_press_at - .000001
            if source_state == 'after_latest_press':
                clock.now += .000010
        return SimpleNamespace(healthy=True, held=source_state != 'released' or clock.now < 1.019)

    mouse = SimpleNamespace(lock=threading.RLock(), snapshot=snapshot)
    presses = []

    class Output:
        def pulse(self):
            presses.append(clock.now)
            return SimpleNamespace(requested_at=clock.now, syn_completed_at=clock.now)

    budget = runtime._cv_budget_seconds([52.378], windows_capture=True)
    assert runtime._service_timer_before_cv(engine, capture, anchor.sequence, mouse=mouse,
                                            output=Output(), cv_budget=budget,
                                            dxgi_frame=anchor)
    assert capture.latest is newer
    if source_state in ('same_source', 'at_latest_press', 'busy_then_ready'):
        expected = {'same_source': plan.press_at, 'at_latest_press': plan.latest_press_at,
                    'busy_then_ready': 1.026}[source_state]
        assert presses == [pytest.approx(expected, abs=.000010)]
        keydown = engine.take_events()[-1]
        assert keydown['kind'] == 'KEYDOWN'
        assert keydown['plan']['timing_mode'] == 'PREDICTED'
        assert keydown['frame_at'] == engine.last_timestamp
        if source_state == 'busy_then_ready':
            assert 0 in waits
    else:
        assert not presses


@pytest.mark.parametrize('same_source', [True, False])
def test_dxgi_runtime_dispatches_due_prediction_after_capture_timeout(
        monkeypatch, tmp_path, capsys, same_source):
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(
        monotonic=lambda: clock.now, perf_counter=lambda: clock.now,
        process_time=lambda: 0.0, sleep=lambda _: None))
    capture = SimpleNamespace(cv=threading.Condition(), stopping=False, error=None,
                              proc=SimpleNamespace(poll=lambda: None), latest=None)
    image = SimpleNamespace(shape=(240, 320, 3))

    def frame(sequence, timestamp, generation=1):
        return Frame(sequence, image, timestamp, timestamp, timestamp, 'test', False,
                     None, '', source_epoch=(41, 1), source_generation=generation)

    # The fake detector makes the production engine plan a press at 1.050.
    # After the timeout, publication can win the race before dispatch checks it.
    def next_frame(after=-1, timeout=1.0):
        if after < 2:
            clock.now = 1.0 + (after + 1) * .020
            capture.latest = frame(after + 1, clock.now)
            return capture.latest
        if capture.latest.sequence == 2:
            assert clock.now + timeout == pytest.approx(1.050)
            clock.now = 1.051
            capture.latest = frame(3, 1.050, generation=1 if same_source else 2)
        else:
            clock.now = 2.0
        return None

    capture.next = next_frame
    capture.prediction_current = lambda sample: sample is capture.latest

    class Detector:
        roi_offset = (0, 0)

        def __init__(self, **_):
            pass

        def measure(self, _image, timestamp, *, center_hint):
            angle = 20.0 + (timestamp - 1.0) * 500
            return Measurement(timestamp, (160, 120), 1.0, Arc(40, 10), None,
                               (Needle(angle, 80, 1, 40, 2),), 'OK')

    monkeypatch.setattr(runtime, 'Detector', Detector)
    result = runtime.run_session(seconds=.1, synthetic=False, fps=50, directory=tmp_path,
        lead_seconds=0, lead_uncertainty=0, recording=False, capture_source='dxgi',
        capture_factory=lambda: nullcontext(capture))

    claims = [event for event in map(json.loads, capsys.readouterr().out.splitlines())
              if event['kind'] == 'PRESS_CLAIM']
    assert result['frames'] == 3
    assert result['physical_presses'] == 0
    assert result['claims'] == len(claims) == int(same_source)
    if same_source:
        assert claims[0]['at'] == pytest.approx(1.051)
        assert claims[0]['plan']['timing_mode'] == 'PREDICTED'


def test_dxgi_planned_press_runs_while_cv_is_still_in_flight(monkeypatch):
    clock = SimpleNamespace(now=1.0)
    frame = Frame(1, None, 1.0, 1.0, 1.0, 'test', False, None, '',
                  source_epoch=(41, 1), source_generation=1)
    capture = SimpleNamespace(cv=threading.Condition(), stopping=False, error=None,
                              proc=SimpleNamespace(poll=lambda: None), latest=frame)
    engine = Engine(lead_seconds=0, lead_uncertainty=0)
    engine.target = Arc(40, 10)
    engine.active = True
    engine.last_timestamp = frame.media_time
    engine.planner.begin(engine.target, 22.5)
    engine.motion.estimate = Motion(1.0, 22.5, 900, .1, 2, 1.0, 12)
    plan = engine.planner.update(engine.motion.estimate, frame_at=1.0, now=1.0)
    future = Future()
    measurement = Measurement(1.0, None, 0, None, None, (), 'NO_PROMPT')
    missed = []

    def sleep(seconds):
        clock.now += max(seconds, .0001)
        if clock.now >= 1.065:
            missed.append(True)
            future.set_result((measurement, 65, 4))

    def pulse():
        assert not future.done()
        assert clock.now == pytest.approx(plan.press_at, abs=.0001)
        future.set_result((measurement, 65, 4))
        return SimpleNamespace(requested_at=clock.now, syn_completed_at=clock.now)

    monkeypatch.setattr(runtime, 'time', SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    mouse = SimpleNamespace(lock=threading.RLock(), snapshot=lambda: SimpleNamespace(held=True, healthy=True))
    result = runtime._wait_for_dxgi_measurement(future, engine, capture, frame, mouse=mouse,
        output=SimpleNamespace(pulse=pulse), stop_at=None)

    assert result == (measurement, 65, 4, True)
    assert not missed
    assert [row['kind'] for row in engine.take_events()] == ['PRESS_CLAIM', 'KEYDOWN']


@pytest.mark.parametrize(('session','generation'),[
    ('20261017_002255_609230',45),('20261018_213036_227179',11),
    ('20261018_213036_227179',22)])
def test_recorded_cv_work_cannot_cover_a_due_white_center_timer(monkeypatch,session,generation):
    from vd.planning import Plan
    trace=next(t for t in TRACES if t['session']==session and t['generation']==generation)
    key=trace['keydown']
    crossed=[(previous,row) for previous,row in zip(trace['rows'],trace['rows'][1:])
        if previous['plan'] is not None
        and row['received_time']<previous['plan']['press_at']<row['decision_time']<key['at']]
    previous,row=crossed[-1]
    clock=SimpleNamespace(now=row['received_time'])
    def monotonic():
        clock.now+=.0000001
        return clock.now
    monkeypatch.setattr(runtime,'time',SimpleNamespace(monotonic=monotonic,
        sleep=lambda seconds:setattr(clock,'now',clock.now+seconds)))
    engine=Engine(lead_seconds=.004)
    plan=Plan(**previous['plan'])
    engine.planner.current=plan
    engine.planner.generation=plan.generation
    engine.planner.version=plan.version
    engine.planner.target=Arc(**key['great'])
    engine.target=engine.planner.target
    capture=SimpleNamespace(cv=threading.Condition(),stopping=False,error=None,
        proc=SimpleNamespace(poll=lambda:None),latest=SimpleNamespace(
            sequence=row['sequence'],media_time=row['media_time'],synthetic=False),
        prediction_current=lambda _:True)
    claims=[]
    class Output:
        def pulse(self):
            claims.append(clock.now)
            return SimpleNamespace(requested_at=clock.now,syn_completed_at=clock.now)
    mouse=SimpleNamespace(lock=threading.RLock(),snapshot=lambda:SimpleNamespace(healthy=True,held=True))
    assert runtime._service_timer_before_cv(engine,capture,row['sequence'],mouse=mouse,
        output=Output(),cv_budget=.012)
    assert len(claims)==1
    assert claims[0]==pytest.approx(plan.press_at,abs=.000002)
    gain=key['at']-claims[0]
    assert gain>.002
    # Timing-only counterfactual: unchanged response and speed, not a game verdict.
    original_angle=(178.1381987631321 if generation==22 else trace['original_landing']['angle'])
    modelled_angle=(original_angle-gain*key['prefire_motion']['speed'])%360
    assert Arc(**key['great']).contains(modelled_angle)


def test_audit_keeps_missing_manifest_sessions_and_rescores_their_evidence(tmp_path):
    from tools.audit_solver_coverage import audit
    trace=next(t for t in TRACES if t['session']=='20261015_183945_769904')
    key=trace['keydown']
    events=[dict(kind='BEGIN',generation=key['generation']),key,trace['end'],
        dict(kind='CV_LANDING',generation=key['generation'],landing=trace['original_landing'])]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    (tmp_path/'frames.jsonl').write_text(''.join(json.dumps(dict(row,
        state=dict(generation=key['generation'],measurement=row['measurement'])))+'\n'
        for row in trace['rows']))
    result=audit(tmp_path,rescore=True)
    assert result['manifest_present'] is False
    assert result['keydowns']==result['checks_with_keydown']==1
    assert result['duplicate_keydowns']==0
    assert result['cv_landings']=={'CV_MISS':1}
    assert result['rescored_cv']['labels']=={'CV_GREAT':1}
    assert result['game_outcomes_measured'] is False
