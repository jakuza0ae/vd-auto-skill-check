"""Single-owner check lifecycle used by live dry-run and recorded replay.

Only observations create generations. Timeouts can end a generation, never
create one. There is no physical input here; callers must dispatch a claimed
plan under the same lock as all engine mutations.
"""
from dataclasses import asdict
import math

from vd.motion import MotionTracker, delta
from vd.planning import Planner, DEFAULT_RESPONSE_DELAY_SECONDS
from vd.vision import same_arc


CHAIN_FAST_SHIFT_DEGREES=25


class Engine:
    def __init__(self, *, lead_seconds=DEFAULT_RESPONSE_DELAY_SECONDS, lead_uncertainty=.020,
                 absence_seconds=.12):
        self.planner=Planner(lead_seconds=lead_seconds,lead_uncertainty=lead_uncertainty)
        self.motion=MotionTracker()
        self.absence_seconds=absence_seconds
        self.active=False
        self.center=None
        self.target=None
        self.good=None
        self.pending=None
        self.pending_at=None
        self.last_visible=None
        self.last_timestamp=None
        self.started_at=None
        self.attempt_at=None
        self.last_reliable_motion_at=None
        self.chain=0
        self.reason='WAITING_FOR_RING'
        self.events=[]

    @property
    def generation(self):
        return self.planner.generation

    def emit(self, kind, at, **values):
        self.events.append(dict(kind=kind,at=at,generation=self.generation,
                                chain=self.chain,**values))

    def take_events(self):
        events,self.events=self.events,[]
        return events

    def cancel(self, reason, now):
        if self.active and not self.planner.fired and reason=='RING_ENDED':
            self.emit('MISSED_END_DIAGNOSTIC',now,
                      end_reason=reason,
                      planner_reason_before_end=self.planner.reason,
                      motion_reason_before_end=self.motion.reason,
                      plan_before_end=(asdict(self.planner.current)
                                       if self.planner.current else None),
                      motion_before_end=(asdict(self.motion.estimate)
                                         if self.motion.estimate else None),
                      occurrence_before_end=(asdict(self.planner.occurrence)
                                             if self.planner.occurrence else None),
                      started_at=self.started_at,
                      last_reliable_motion_at=self.last_reliable_motion_at,
                      last_visible=self.last_visible)
        self.planner.invalidate(reason)
        self.reason=reason
        if self.active:
            self.emit('END',now,reason=reason,pressed=self.planner.fired)
        self.active=False
        self.center=None
        self.target=None
        self.good=None
        self.pending=None
        self.pending_at=None
        self.last_visible=None
        self.last_reliable_motion_at=None
        self.motion.reset()
        self.attempt_at=None
        self.chain=0

    def observe(self, m, *, now, held=True):
        if not math.isfinite(now):
            raise ValueError('Non-finite decision time')
        if self.last_timestamp is not None and m.timestamp<=self.last_timestamp:
            self.motion.reset_fit()
            self.planner.invalidate('NONINCREASING_TIME')
            self.reason='NONINCREASING_TIME'
            return
        self.last_timestamp=m.timestamp
        if not held:
            self.cancel('LMB_RELEASED',now)
            return
        if m.timestamp>now+.002 or now-m.timestamp>self.planner.max_age:
            self.motion.reset_fit()
            self.planner.invalidate('STALE_FRAME')
            self.reason='STALE_FRAME'
            return
        visible=m.prompt_score>=.80 and m.center is not None and m.great is not None
        if not visible:
            self.pending=None
            self.pending_at=None
            self.motion.update(m.timestamp,())
            self.planner.invalidate('RING_NOT_VISIBLE')
            self.reason='RING_NOT_VISIBLE'
            if self.active and not self.planner.fired:
                self.planner.attempt_now(frame_at=m.timestamp,now=now,reason='RING_DISAPPEARING')
                self.reason='LAST_CHANCE'
                return
            if self.last_visible is not None and m.timestamp-self.last_visible>=self.absence_seconds:
                self.cancel('RING_ENDED',now)
            return
        self.last_visible=m.timestamp
        shifted=(not self.active or not same_arc(self.target,m.great)
                 or math.dist(self.center,m.center)>3)
        if shifted:
            if self.active and not self.planner.fired:
                self.planner.attempt_now(frame_at=m.timestamp,now=now,reason='GEOMETRY_CHANGED')
                self.reason='LAST_CHANCE'
                return
            self.reason='CONFIRMING_GEOMETRY'
            previous_target=self.target
            previous_occurrence=self.planner.occurrence
            if self.planner.current is not None:
                self.planner.invalidate('GEOMETRY_PENDING')
            continuing_chain=(self.active and self.planner.fired
                              and math.dist(self.center,m.center)<=3)
            large_chain_shift=(continuing_chain and previous_target is not None and
                               abs(delta(m.great.start,previous_target.start))
                               >=CHAIN_FAST_SHIFT_DEGREES)
            if not continuing_chain:
                self.motion.reset_fit()
            pending_matches=(self.pending is not None
                and same_arc(self.pending[0],m.great)
                and math.dist(self.pending[1],m.center)<=3)
            if not large_chain_shift:
                if (not pending_matches or self.pending_at is None
                        or m.timestamp-self.pending_at>self.motion.max_gap):
                    first=max(m.candidates,key=lambda c:c.contrast,default=None)
                    self.pending=(m.great,m.center,first.angle if first else None)
                    self.pending_at=m.timestamp
                    return
                if m.timestamp-self.pending_at<.012:
                    return
            strongest=max(m.candidates,key=lambda c:c.contrast,default=None)
            prior_fired=self.active and self.planner.fired
            if self.active:
                self.emit('END',now,reason='GEOMETRY_CHANGED',pressed=self.planner.fired)
            self.chain=self.chain+1 if prior_fired else 1
            self.target=m.great
            self.good=m.good
            self.center=m.center
            initial=(self.pending[2] if pending_matches and self.pending[2] is not None else
                     strongest.angle if strongest is not None else m.great.center)
            occurrence_start=None
            if continuing_chain:
                phase_floor=(self.motion.last_unwrapped
                             if self.motion.last_unwrapped is not None
                             else self.motion.unwrap_floor)
                if phase_floor is not None:
                    initial=phase_floor+delta(initial,phase_floor)
                if previous_target is not None and previous_occurrence is not None:
                    forward_shift=(m.great.start-previous_target.start)%360
                    occurrence_start=previous_occurrence.great_start+forward_shift
            self.planner.begin(m.great,initial,m.good,
                               occurrence_start=occurrence_start)
            if not continuing_chain:
                self.motion.reset()
                self.motion.unwrap_floor=initial
                self.last_reliable_motion_at=None
            self.active=True
            self.started_at=m.timestamp
            self.attempt_at=now+.300
            self.pending=None
            self.pending_at=None
            self.emit('BEGIN',now,target=asdict(m.great))
        else:
            self.pending=None
            self.pending_at=None
        self.motion.white_width=self.target.width
        estimate=self.motion.update(m.timestamp,m.candidates)
        if estimate is not None:
            self.last_reliable_motion_at=estimate.last_motion_at
        if self.planner.fired:
            self.reason='OBSERVING_AFTER_PRESS'
            return
        plan=self.planner.update(estimate,frame_at=m.timestamp,now=now)
        self.reason=self.planner.reason if estimate is not None else self.motion.reason
        if plan:
            self.attempt_at=plan.press_at
            self.reason=plan.timing_mode
        elif self.attempt_at is not None and now>=self.attempt_at:
            self.planner.attempt_now(frame_at=m.timestamp,now=now,reason=self.motion.reason)
            self.reason='LAST_CHANCE'

    def poll(self, now, *, held=True, capture_alive=True):
        if not held or not capture_alive:
            self.cancel('LMB_RELEASED' if not held else 'CAPTURE_LOST',now)
            return None
        plan=self.planner.current
        if (plan is None and self.active and not self.planner.fired
                and self.attempt_at is not None and now>=self.attempt_at
                and self.last_timestamp is not None):
            plan=self.planner.attempt_now(frame_at=self.last_timestamp,now=now,
                reason='TRACKING_UNAVAILABLE')
        if plan is not None and now>plan.latest_press_at and now<=plan.valid_until:
            frame_at=self.last_timestamp if self.last_timestamp is not None else plan.valid_until-self.planner.max_age
            plan=self.planner.update(self.motion.estimate,frame_at=frame_at,now=now)
            if plan is None:
                plan=self.planner.attempt_now(frame_at=frame_at,now=now,reason='DEADLINE_MISSED')
            if plan is not None:
                self.attempt_at=plan.press_at
        if plan is not None and now>plan.valid_until:
            self.planner.invalidate('OBSERVATION_EXPIRED')
            self.reason='OBSERVATION_EXPIRED'
            return None
        if plan is None:
            return None
        if not self.planner.claim(plan,now=now,held=held,capture_alive=capture_alive):
            if self.planner.current is None:
                self.reason=self.planner.reason
            return None
        self.emit('PRESS_CLAIM',now,plan=asdict(plan))
        self.reason='OBSERVING_AFTER_PRESS'
        return plan

    def snapshot(self):
        return dict(generation=self.generation,chain=self.chain,active=self.active,
                    fallback_press_at=self.attempt_at,
                    lead_seconds=self.planner.lead,lead_uncertainty=self.planner.lead_uncertainty,
                    reason=self.reason,motion_reason=self.motion.reason,
                    motion=asdict(self.motion.estimate) if self.motion.estimate else None,
                    plan=asdict(self.planner.current) if self.planner.current else None)
