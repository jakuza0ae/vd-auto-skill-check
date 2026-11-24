"""Robust phase and speed from timestamped independent measurements.

No session speed prior and no key dispatch. A frame discontinuity invalidates
motion. Velocity uses a robust fit; phase follows the latest accepted measurement.
"""
from collections import deque
from dataclasses import dataclass
import math
import statistics


def delta(a,b):
    return (a-b+180)%360-180


@dataclass(frozen=True)
class Motion:
    at: float
    phase: float
    speed: float
    residual: float
    speed_scatter: float
    last_motion_at: float
    samples: int
    latest_innovation: float = 0.0

    def phase_at(self, now):
        return self.phase+self.speed*(now-self.at)


class MotionTracker:
    MAX_SPEED=1800.0
    SPEED_EPS=1e-4

    def __init__(self, window=.22, max_gap=.080):
        self.window=window
        self.max_gap=max_gap
        self.points=deque(maxlen=64)
        self.last_frame=None
        self.last_angle=None
        self.last_unwrapped=None
        self.last_accepted_at=None
        self.last_observed_motion_at=None
        self.unwrap_floor=None
        self.estimate=None
        self.reason='EMPTY'
        self.capture_intervals=deque(maxlen=8)
        self.motion_intervals=deque(maxlen=8)
        self.white_width=10.0

    def reset(self):
        self.points.clear()
        self.capture_intervals.clear()
        self.motion_intervals.clear()
        self.last_observed_motion_at=None
        self.last_frame=None
        self.last_angle=None
        self.last_unwrapped=None
        self.last_accepted_at=None
        self.unwrap_floor=None
        self.estimate=None
        self.reason='EMPTY'

    def reset_fit(self):
        """Discard velocity samples while retaining the observed revolution."""
        floor=self.last_unwrapped if self.last_unwrapped is not None else self.unwrap_floor
        motion_at=self.last_observed_motion_at
        self.reset()
        self.unwrap_floor=floor
        self.last_observed_motion_at=motion_at

    def stall_after(self):
        capture=statistics.median(self.capture_intervals) if self.capture_intervals else .020
        if len(self.motion_intervals)<3 or self.estimate is None:
            return min(self.max_gap,2*capture)
        moving=statistics.median(self.motion_intervals)
        dwell=self.white_width/self.estimate.speed
        return min(self.max_gap,moving+min(capture/2,dwell/2))

    def update(self, timestamp, candidates):
        if not math.isfinite(timestamp):
            raise ValueError('Non-finite timestamp')
        if self.last_frame is not None and timestamp<=self.last_frame:
            previous=self.last_frame
            self.reset_fit()
            self.last_frame=previous
            self.reason='NONINCREASING_TIME'
            return None
        if self.last_frame is not None and timestamp-self.last_frame>self.max_gap:
            self.reset_fit()
            self.reason='FRAME_GAP'
        if self.last_accepted_at is not None and timestamp-self.last_accepted_at>self.max_gap:
            # Delivery can continue while the needle disappears or is frozen.
            # Reacquire from scratch instead of joining disconnected tracks.
            self.reset_fit()
            self.reason='TRACK_GAP'
        if self.last_frame is not None:
            self.capture_intervals.append(timestamp-self.last_frame)
        self.last_frame=timestamp
        while self.points and timestamp-self.points[0][0]>self.window:
            self.points.popleft()
        if not candidates:
            self.reset_fit()
            self.last_frame=timestamp
            self.reason='NO_NEEDLE'
            return None
        if self.estimate:
            expected=self.estimate.phase_at(timestamp)%360
            chosen=min(candidates,key=lambda c:abs(delta(c.angle,expected)))
        elif self.last_angle is not None:
            chosen=min(candidates,key=lambda c:abs(delta(c.angle,self.last_angle)))
        else:
            chosen=max(candidates,key=lambda c:c.contrast)
        angle=chosen.angle
        step=delta(angle,self.last_angle) if self.last_angle is not None else None
        elapsed=(timestamp-self.last_accepted_at if self.last_accepted_at is not None else self.max_gap)
        frame_period=(statistics.median(self.capture_intervals)
                      if self.capture_intervals else 0)
        step_noise=(self.estimate.speed if self.estimate else self.MAX_SPEED)*frame_period/2
        max_step=min(179.0,self.MAX_SPEED*elapsed+max(2,step_noise))
        if step is not None and abs(step)<.5:
            if (self.last_observed_motion_at is not None
                    and timestamp-self.last_observed_motion_at>=self.stall_after()-1e-9):
                self.reset_fit()
                self.last_frame=timestamp
                self.reason='STALLED'
                return None
            self.reason='REPEATED_ANGLE'
            return self.estimate
        if self.estimate is not None:
            innovation=abs(delta(angle,self.estimate.phase_at(timestamp)%360))
            frame_period=(statistics.median(self.capture_intervals)
                          if self.capture_intervals else 0)
            phase_tolerance=max(5.0,3*self.estimate.residual,
                                self.estimate.speed*frame_period)
            if innovation>phase_tolerance:
                floor=self.last_unwrapped
                self.reset_fit()
                self.last_frame=timestamp
                if floor is not None and step is not None and .5<step<=max_step:
                    unwrapped=floor+step
                    self.last_angle=angle
                    self.last_unwrapped=unwrapped
                    self.last_accepted_at=timestamp
                    self.last_observed_motion_at=timestamp
                    self.unwrap_floor=unwrapped
                    self.points.append((timestamp,unwrapped))
                self.reason='PHASE_OUTLIER'
                return None
        if self.last_angle is None:
            # Reacquisition drops the fit, not the turn already observed.
            # Otherwise a post-wrap frame could turn a passed target into an
            # apparent target one rotation in the future.
            if self.unwrap_floor is None:
                unwrapped=angle
            else:
                step=delta(angle,self.unwrap_floor)
                if abs(step)<.5:
                    self.reason='WAITING_FOR_MOVEMENT'
                    return None
                # The same <=2° backward pixel noise is tolerated during
                # reacquisition as during an established track.
                unwrapped=self.unwrap_floor+(step if step>=-2 else step+360)
                if .5<step<=max_step:
                    self.last_observed_motion_at=timestamp
        else:
            step=delta(angle,self.last_angle)
            if step<-2 or step>max_step:
                self.reset_fit()
                self.last_frame=timestamp
                self.reason='ANGLE_DISCONTINUITY'
                return None
            if step>.5 and self.last_observed_motion_at is not None:
                self.motion_intervals.append(timestamp-self.last_observed_motion_at)
            unwrapped=self.last_unwrapped+step
            if step>.5:
                self.last_observed_motion_at=timestamp
        self.last_angle=angle
        self.last_unwrapped=unwrapped
        self.last_accepted_at=timestamp
        self.points.append((timestamp,unwrapped))
        if len(self.points)<4 or self.points[-1][0]-self.points[0][0]<.04:
            if (len(self.points)>=2 and self.points[-1][0]-self.points[0][0]>=.012
                    and self.points[-1][1]-self.points[0][1]>=4):
                span=self.points[-1][0]-self.points[0][0]
                speed=(self.points[-1][1]-self.points[0][1])/span
                if 20-self.SPEED_EPS<=speed<=self.MAX_SPEED+self.SPEED_EPS:
                    speed=min(self.MAX_SPEED,max(20.0,speed))
                    self.estimate=Motion(timestamp,unwrapped,speed,1.5,
                                         max(80,.2*speed),timestamp,len(self.points))
                    self.reason='EARLY_MOTION'
                    return self.estimate
            self.estimate=None
            self.reason='MEASURING_MOTION'
            return None
        points=list(self.points)
        slopes=[(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(points)
                for b in points[i+1:] if b[0]-a[0]>=.025]
        if not slopes:
            return None
        speed=statistics.median(slopes)
        if points[-1][1]-points[0][1]<4:
            self.estimate=None
            self.reason='MEASURING_MOTION'
            return None
        # Center time avoids precision loss from large monotonic timestamps.
        scatter=1.4826*statistics.median(abs(x-speed) for x in slopes)
        # A fit near the physical limit can straddle it because frame timestamps
        # quantize otherwise coherent motion. Constant actual overspeed has no
        # such scatter and remains invalid.
        speed_tolerance=max(self.SPEED_EPS,min(.20*self.MAX_SPEED,scatter))
        if not 20-speed_tolerance<=speed<=self.MAX_SPEED+speed_tolerance:
            self.reset_fit()
            self.last_frame=timestamp
            self.reason='INVALID_SPEED'
            return None
        speed=min(self.MAX_SPEED,max(20.0,speed))
        offsets=[a-speed*(t-timestamp) for t,a in points]
        phase=statistics.median(offsets)
        residual=statistics.median(abs(x-phase) for x in offsets)
        scatter=1.4826*statistics.median(abs(x-speed) for x in slopes)
        frame_period=(statistics.median(self.capture_intervals)
                      if self.capture_intervals else 0)
        residual_limit=max(3.0,speed*frame_period/2)
        if residual>residual_limit or scatter>max(45,.20*speed):
            self.reset_fit()
            self.last_frame=timestamp
            self.reason='INCOHERENT_MOTION'
            return None
        self.estimate=Motion(timestamp,unwrapped,speed,residual,scatter,timestamp,len(points),unwrapped-phase)
        self.reason='MEASURED'
        return self.estimate
