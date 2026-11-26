"""Final dispatch gate for the single-owner runtime.

Locks publication of capture and mouse state only across the final check and
keydown. CV and recording must never run under these locks. This bounds races
with already published observations; it cannot make OS event delivery atomic.
"""
from contextlib import nullcontext
from dataclasses import asdict
import time

from .input_state import InputDisarmed


def dispatch(engine, capture, sequence, *, mouse=None, output=None, clock=time.monotonic,
             same_source_as=None, predicted_only=False):
    if output is not None and mouse is None:
        raise ValueError('Physical input requires a mouse monitor')
    with capture.cv:
        now=clock()
        if capture.stopping or capture.error or capture.proc.poll() is not None:
            engine.cancel('CAPTURE_LOST',now)
            return None
        frame=capture.latest
        prediction_current=getattr(capture,'prediction_current',None)
        plan=engine.planner.current
        # A validated DXGI timer may outlive the publication of its frame.
        # New pixels only confirm the capture source, never change the plan.
        same_source_timer=(same_source_as is not None and frame is not None
            and frame.sequence>sequence and same_source_as.sequence==sequence
            and same_source_as.source_epoch is not None
            and (frame.source_epoch,frame.source_generation)==
                (same_source_as.source_epoch,same_source_as.source_generation)
            and plan is not None and plan.timing_mode=='PREDICTED'
            and plan.press_at<=now<=min(plan.valid_until,plan.latest_press_at))
        if frame is None or (frame.sequence!=sequence and not same_source_timer):
            if engine.planner.current is not None and now>engine.planner.current.valid_until:
                engine.planner.invalidate('OBSERVATION_EXPIRED')
            engine.reason='NEW_FRAME_PENDING'
            return None
        def prediction_ready():
            if prediction_current is None or prediction_current(frame):
                return True
            busy=getattr(capture,'prediction_busy',lambda:False)()
            engine.reason='CAPTURE_WRITING' if busy else 'NEW_FRAME_PENDING'
            return False
        if not prediction_ready():
            return None
        if output is not None and (frame.synthetic or
                                   (same_source_timer and same_source_as.synthetic)):
            engine.cancel('SYNTHETIC_INPUT_FORBIDDEN',now)
            raise ValueError('Physical input is forbidden for synthetic capture')
        if frame.media_time is None or not -.002<=now-frame.media_time<=engine.planner.max_age:
            engine.planner.invalidate('STALE_FRAME')
            engine.reason='STALE_FRAME'
            return None
        with mouse.lock if mouse is not None else nullcontext():
            state=mouse.snapshot() if mouse is not None else None
            if state is not None and not state.healthy:
                engine.cancel('MOUSE_UNAVAILABLE',clock())
                return None
            held=state.held if state is not None else True
            if not prediction_ready():
                return None
            claim_at=clock()
            if (same_source_timer or predicted_only) and (plan is None or
                    plan.timing_mode!='PREDICTED' or claim_at>min(plan.valid_until,plan.latest_press_at)):
                return None
            if claim_at-frame.media_time>engine.planner.max_age:
                engine.planner.invalidate('STALE_FRAME')
                engine.reason='STALE_FRAME'
                return None
            plan=engine.poll(claim_at,held=held,capture_alive=True)
            if plan is None:
                return None
            if output is not None:
                try:
                    keydown=output.pulse()
                except InputDisarmed as exc:
                    engine.emit('INPUT_DISARMED',clock(),error=str(exc),plan=asdict(plan))
                    engine.cancel('INPUT_DISARMED',clock())
                    return None
                except BaseException as exc:
                    engine.emit('INPUT_FAILED',clock(),error=str(exc),plan=asdict(plan))
                    engine.cancel('INPUT_FAILED',clock())
                    raise
                frame_at=frame.media_time
                if same_source_as is not None and engine.last_timestamp is not None:
                    frame_at=engine.last_timestamp
                engine.emit('KEYDOWN',keydown.syn_completed_at,
                    claimed_at=claim_at,requested_at=keydown.requested_at,plan=asdict(plan),
                    frame_at=frame_at,
                    prefire_motion=asdict(engine.motion.estimate) if engine.motion.estimate else None,
                    motion_diagnostics=dict(
                        reason=engine.motion.reason,
                        samples=len(engine.motion.points),
                        last_observed_motion_at=engine.motion.last_observed_motion_at,
                        check_age_ms=(None if engine.started_at is None else
                                      (frame_at-engine.started_at)*1000)),
                    great=asdict(engine.planner.target),
                    good=asdict(engine.good) if engine.good else None,
                    dispatch_lag_ms=(keydown.syn_completed_at-plan.intended_press_at)*1000,
                    outside_target_window=(None if plan.timing_mode!='PREDICTED' else
                        bool(keydown.syn_completed_at>plan.latest_press_at)))
            return plan
