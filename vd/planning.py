"""Pure scheduling contracts. This module never opens an input device.

A caller owns generations and must invalidate on capture loss, lifecycle change
or LMB release. Each generation owns one immutable unwrapped success occurrence;
update() always aims at GREAT; a missed deadline triggers one immediate attempt.
"""
from dataclasses import dataclass
import math

from vd.motion import Motion
from vd.vision import Arc


DEFAULT_RESPONSE_DELAY_SECONDS=.004


@dataclass(frozen=True)
class SuccessOccurrence:
    great_start: float
    great_end: float
    great_center: float
    good_start: float | None
    good_end: float | None

    @property
    def success_end(self):
        return self.good_end if self.good_end is not None else self.great_end


@dataclass(frozen=True)
class Plan:
    generation: int
    version: int
    press_at: float
    intended_press_at: float
    valid_until: float
    target_phase: float
    uncertainty_degrees: float | None
    latest_press_at: float
    target_grade: str
    target_window_start: float
    target_window_width: float
    timing_mode: str = 'PREDICTED'
    fallback_reason: str | None = None


class Planner:
    EPS=.25
    def __init__(self, *, lead_seconds, lead_uncertainty=.015, max_age=.080):
        if not all(math.isfinite(v) for v in (lead_seconds,lead_uncertainty,max_age)):
            raise ValueError('Non-finite timing configuration')
        if lead_seconds<0 or lead_uncertainty<0 or max_age<=0:
            raise ValueError('Invalid timing configuration')
        self.lead=lead_seconds
        self.lead_uncertainty=lead_uncertainty
        self.max_age=max_age
        self.generation=0
        self.version=0
        self.target_phase=None
        self.target=None
        self.good=None
        self.occurrence=None
        self.fired=False
        self.current=None
        self.reason='NO_GENERATION'

    def _occurrence_from_start(self, target, good, start):
        great_start=float(start)
        great_end=great_start+target.width
        great_center=great_start+target.width/2
        good_start=good_end=None
        if good is not None:
            gap=(good.start-(target.start+target.width))%360
            if gap<=8:
                good_start=great_end+gap
                good_end=good_start+good.width
        return SuccessOccurrence(great_start,great_end,great_center,
                                 good_start,good_end)

    def begin(self, target: Arc, initial_phase: float, good: Arc | None = None, *,
              occurrence_start: float | None = None, allow_trailing_good=None):
        if not (math.isfinite(initial_phase) and math.isfinite(target.start)
                and math.isfinite(target.width) and 0<target.width<180):
            raise ValueError('Invalid generation geometry')
        if good is not None and not (math.isfinite(good.start)
                and math.isfinite(good.width) and 0<good.width<180):
            raise ValueError('Invalid GOOD geometry')
        if occurrence_start is not None and not math.isfinite(occurrence_start):
            raise ValueError('Invalid occurrence anchor')
        self.invalidate('NEW_GENERATION')
        self.generation+=1
        self.fired=False
        self.target=target
        self.good=good

        if occurrence_start is None:
            next_start=initial_phase+(target.start-initial_phase)%360
            previous=self._occurrence_from_start(target,good,next_start-360)
            if previous.great_start-self.EPS<=initial_phase<=previous.success_end+self.EPS:
                self.occurrence=previous
            else:
                self.occurrence=self._occurrence_from_start(target,good,next_start)
        else:
            self.occurrence=self._occurrence_from_start(target,good,occurrence_start)
        self.target_phase=self.occurrence.great_center

    def invalidate(self, reason):
        self.version+=1
        self.current=None
        self.reason=reason

    def _uncertainty(self, motion: Motion, phase: float):
        horizon=max(0,(phase-motion.phase)/motion.speed)
        return (max(.75,3*motion.residual,abs(motion.latest_innovation))+motion.speed_scatter*horizon
                +motion.speed*self.lead_uncertainty)

    def _make_plan(self, motion, *, frame_at, now, aim, window_start,
                   window_end, uncertainty):
        valid_until=min(frame_at,motion.last_motion_at)+self.max_age
        intended_press_at=motion.at+(aim-motion.phase)/motion.speed-self.lead
        latest_press_at=motion.at+(window_end-motion.phase)/motion.speed-self.lead
        press_at=max(now,intended_press_at)
        self.version+=1
        self.current=Plan(self.generation,self.version,press_at,intended_press_at,valid_until,
                          aim,uncertainty,latest_press_at,'GREAT',
                          window_start,window_end-window_start)
        self.reason='PLANNED'
        return self.current

    def attempt_now(self, *, frame_at, now, reason):
        if self.target is None or self.occurrence is None or self.fired:
            return None
        if (not math.isfinite(now) or not math.isfinite(frame_at)
                or not -.002<=now-frame_at<=self.max_age):
            self.invalidate('STALE_OBSERVATION')
            return None
        self.version+=1
        phase=self.occurrence.great_center
        self.current=Plan(self.generation,self.version,now,now,frame_at+self.max_age,
            phase,None,frame_at+self.max_age,'GREAT',self.occurrence.great_start,
            self.occurrence.great_end-self.occurrence.great_start,'LAST_CHANCE',reason)
        self.reason='LAST_CHANCE_'+reason
        return self.current

    def update(self, motion: Motion | None, *, frame_at: float, now: float):
        existing=self.current
        if self.target is None or self.occurrence is None or self.fired:
            return None
        if existing is not None and (now>existing.valid_until
                                     or existing.press_at>existing.valid_until):
            self.invalidate('OBSERVATION_EXPIRED' if now>existing.valid_until
                            else 'PLAN_NOT_DISPATCHABLE')
            existing=None
        if motion is None:
            self.invalidate('NO_MOTION')
            return None
        if not all(math.isfinite(v) for v in (frame_at,now,motion.at,motion.phase,
                motion.speed,motion.residual,motion.speed_scatter,motion.last_motion_at,
                motion.latest_innovation)):
            self.invalidate('INVALID_TIMING')
            return None
        valid_until=min(frame_at,motion.last_motion_at)+self.max_age
        if frame_at>now+.002 or motion.at>now+.002 or now>valid_until:
            self.invalidate('STALE_OBSERVATION')
            return None
        if motion.speed<=0 or motion.residual<0 or motion.speed_scatter<0:
            self.invalidate('INVALID_MOTION')
            return None

        occ=self.occurrence
        effect_phase=motion.phase_at(now+self.lead)
        if effect_phase>=occ.great_end:
            return self.attempt_now(frame_at=frame_at,now=now,reason='GREAT_PASSED')

        uncertainty=self._uncertainty(motion,occ.great_center)
        if effect_phase>occ.great_center+1e-9:
            return self.attempt_now(frame_at=frame_at,now=now,reason='CENTER_PASSED')
        margin=min(1.0,self.target.width*.15)
        start=max(occ.great_start+margin,effect_phase)
        return self._make_plan(motion,frame_at=frame_at,now=now,aim=occ.great_center,
            window_start=start,window_end=occ.great_end-margin,
            uncertainty=uncertainty)

    def claim(self, plan: Plan, *, now: float, held: bool, capture_alive: bool):
        """Consume once immediately before dispatch, under the runtime lock."""
        if (not math.isfinite(now) or not held or not capture_alive or self.fired
                or self.current is not plan or plan.generation!=self.generation
                or plan.version!=self.version or now<plan.press_at
                or now>plan.valid_until or now>plan.latest_press_at):
            return False
        self.fired=True
        self.invalidate('CLAIMED')
        return True
