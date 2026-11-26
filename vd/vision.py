"""Independent per-image measurements; no wall clock and no hidden tracking.

Geometry is calibrated for the user's 1080p game UI. Red candidates require
radial support and local angular contrast, not just a high red-channel mean.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import math

import cv2
import numpy as np

from .image_io import read_image


def configure_realtime_opencv():
    """Bound OpenCV's worker pool inside the latency-sensitive solver process."""
    cv2.setNumThreads(1)


configure_realtime_opencv()


@dataclass(frozen=True)
class Arc:
    start: float
    width: float

    @property
    def center(self):
        return (self.start + self.width / 2) % 360

    def contains(self, angle):
        return (angle - self.start) % 360 <= self.width


@dataclass(frozen=True)
class Needle:
    angle: float
    strength: float
    radial_support: float
    contrast: float
    angular_width: float


@dataclass(frozen=True)
class Measurement:
    timestamp: float
    center: tuple[float, float] | None
    prompt_score: float
    great: Arc | None
    good: Arc | None
    candidates: tuple[Needle, ...]
    reason: str
    great_source: str = 'MEASURED'


def same_arc(a, b):
    return abs((a.center-b.center+180)%360-180)<=3 and abs(a.width-b.width)<=3


def retained_target(m, *, great, good, center):
    """Retain an observed arc only when a thin needle occludes its boundary.

    This is explicitly tracked geometry, not a new per-image measurement.
    Missing white also requires the unchanged black arc as independent evidence.
    """
    if (great is None or center is None or m.center is None or m.prompt_score<.80
            or math.dist(center,m.center)>3 or len(m.candidates)!=1):
        return m
    needle=m.candidates[0]
    if not great.contains(needle.angle):
        return m
    if m.great is None:
        if good is None or m.good is None or not same_arc(good,m.good):
            return m
    else:
        if same_arc(great,m.great):
            return m
        offset=(m.great.start-great.start+180)%360-180
        end=offset+m.great.width
        if not (-1.5<=offset and end<=great.width+1.5 and m.great.width<great.width):
            return m
        start_visible=abs(offset)<=1.5
        end_visible=abs(end-great.width)<=1.5
        boundary=(m.great.start+m.great.width) if start_visible else m.great.start
        near_needle=abs((boundary-needle.angle+180)%360-180)<=needle.angular_width/2+2.5
        if not (start_visible or end_visible) or not near_needle:
            return m
    return replace(m,great=great,great_source='TRACKED_NEEDLE_OCCLUSION',reason='OK')


def arcs_from_score(score, minimum, maximum):
    """Circular positive runs with interpolated zero crossings, in degrees."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    mask = score > 0
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    result = []
    for start in starts:
        length = 1
        while length < n and mask[(start + length) % n]:
            length += 1
        end = (start + length - 1) % n
        before = score[(start - 1) % n]
        first = score[start]
        last = score[end]
        after = score[(end + 1) % n]
        a = start - 1 + (-before / (first - before))
        b = start + length - 1 + last / (last - after)
        width = (b - a) * 360 / n
        if minimum <= width <= maximum:
            result.append(Arc(float(a * 360 / n % 360), float(width)))
    return result


def _half_size(image):
    height, width = image.shape
    return cv2.resize(image, (round(width * .5), round(height * .5)),
                      interpolation=cv2.INTER_AREA)


class _PromptFrame:
    def __init__(self, frame, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Frame timestamp must be finite')
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.half_gray = None

    def reduced(self):
        if self.half_gray is None:
            self.half_gray = _half_size(self.gray)
        return self.half_gray


class Detector:
    def __init__(self, template=None, ring_radius=66.0, *, roi_offset=(0, 0),
                 ui_scale=1.0, acquisition_center=None):
        assets = Path(__file__).resolve().parents[1] / 'assets'
        # Both prompt canvases are aligned to the ring center. The update's
        # jagged key outline coexists with the legacy Space icon in replays.
        paths = (Path(template),) if template else (
            assets / 'space_template.png', assets / 'space_template_20261024.png')
        self.ui_scale = float(ui_scale)
        if not math.isfinite(self.ui_scale) or self.ui_scale <= 0:
            raise ValueError('UI scale must be finite and positive')
        templates = []
        for path in paths:
            prompt = read_image(path, cv2.IMREAD_GRAYSCALE)
            if prompt is None:
                raise ValueError(f'Cannot load prompt template: {path}')
            if self.ui_scale != 1:
                height, width = prompt.shape
                prompt = cv2.resize(prompt,
                    (round(width*self.ui_scale),round(height*self.ui_scale)),
                    interpolation=cv2.INTER_AREA if self.ui_scale<1 else cv2.INTER_LINEAR)
            templates.append(prompt)
        self.templates = tuple(templates)
        self.template = self.templates[0]
        self._half_templates = tuple(_half_size(prompt) if min(prompt.shape) > 1 else None
                                     for prompt in self.templates)
        self.ring_radius = float(ring_radius)*self.ui_scale
        self.roi_offset = tuple(roi_offset)
        offset_x, offset_y = self.roi_offset
        self._acquisition_centers = ((tuple(acquisition_center),) if acquisition_center is not None
                                     else ((160-offset_x,162.5-offset_y),
                                           (170-offset_x,82.5-offset_y)))
        self.angles = np.arange(720, dtype=np.float32) * (math.pi / 360)
        self._cos=np.cos(self.angles)[:,None]
        self._sin=np.sin(self.angles)[:,None]
        self._ring_radii=np.linspace(self.ring_radius-3*self.ui_scale,
                                     self.ring_radius+3*self.ui_scale,9,dtype=np.float32)
        self._needle_radii=np.linspace(24*self.ui_scale,61*self.ui_scale,32,dtype=np.float32)
        self._sample_maps={}

    def _sample(self, frame, center, radii):
        r = np.asarray(radii, np.float32)[None, :]
        key=(tuple(center),r.shape,r.tobytes())
        maps=self._sample_maps.get(key)
        if maps is None:
            maps=(center[0]+self._cos*r,center[1]+self._sin*r)
            if len(self._sample_maps)>=8:
                self._sample_maps.clear()
            self._sample_maps[key]=maps
        x,y=maps
        return cv2.remap(frame, x, y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT).astype(np.float32)

    def _search_full(self, gray, small, template, small_template):
        height,width=gray.shape
        th,tw=template.shape
        result=cv2.matchTemplate(small,small_template,cv2.TM_CCOEFF_NORMED)
        _,coarse,_,location=cv2.minMaxLoc(result)
        if coarse<.65:
            return float(coarse),None
        left=round(location[0]*width/small.shape[1])
        top=round(location[1]*height/small.shape[0])
        x0=max(0,left-4);y0=max(0,top-4)
        x1=min(width,left+tw+5);y1=min(height,top+th+5)
        result=cv2.matchTemplate(gray[y0:y1,x0:x1],template,
                                 cv2.TM_CCOEFF_NORMED)
        _,score,_,location=cv2.minMaxLoc(result)
        return float(score),(x0+location[0],y0+location[1])

    def measure(self, frame, timestamp, *, center_hint=None):
        return self._measure(frame, timestamp, _PromptFrame(frame, timestamp), center_hint)

    def _measure(self, frame, timestamp, prompt, center_hint):
        gray = prompt.gray
        th, tw = self.template.shape
        h, w = gray.shape
        if h < th or w < tw:
            raise ValueError('ROI is smaller than prompt template')
        score=-1.0
        prompt_center=None
        # Search the observed positions for this backend before looking across the ROI.
        for template in self.templates:
            th,tw=template.shape
            for candidate in dict.fromkeys((center_hint,*self._acquisition_centers)):
                if candidate is None:
                    continue
                left=round(candidate[0]-tw/2)
                top=round(candidate[1]-th/2)
                x0=max(0,left-5);y0=max(0,top-5)
                x1=min(w,left+tw+5);y1=min(h,top+th+5)
                if x1-x0<tw or y1-y0<th:
                    continue
                result=cv2.matchTemplate(gray[y0:y1,x0:x1],template,
                                         cv2.TM_CCOEFF_NORMED)
                _,value,_,local=cv2.minMaxLoc(result)
                if value>score:
                    score=value
                    prompt_center=(x0+local[0]+tw/2,y0+local[1]+th/2)
                if value>=.90 and (candidate==center_hint or len(self._acquisition_centers)==1):
                    break
            if score>=.90:
                break
        if score < .80:
            small = prompt.reduced()
            for template, small_template in zip(self.templates, self._half_templates):
                if small_template is None:
                    small_template = _half_size(template)
                value,local=self._search_full(gray,small,template,small_template)
                if local is not None and value>score:
                    th,tw=template.shape
                    score=value
                    prompt_center=(local[0]+tw/2,local[1]+th/2)
                if score>=.90:
                    break
        center = prompt_center if score >= .80 else center_hint
        if center is None:
            return Measurement(timestamp, None, float(score), None, None, (), 'NO_PROMPT')
        cx, cy = center
        radius = self.ring_radius + 4*self.ui_scale
        if not (radius <= cx < w-radius and radius <= cy < h-radius):
            return Measurement(timestamp, center, float(score), None, None, (), 'RING_OUTSIDE_ROI')
        ring = self._sample(frame,center,self._ring_radii)
        levels = ring.mean(axis=1)
        intensity = levels.mean(axis=1)
        background = float(np.median(intensity))
        white_threshold = min(185, max(160, background+40))
        white_score = np.minimum(intensity-white_threshold, levels.min(axis=1)-150)
        black_threshold = max(42, min(55, background-40))
        whites = arcs_from_score(white_score, 5, 16)
        blacks = arcs_from_score(black_threshold-intensity, 18, 65)
        pairs = [(a,b) for a in whites for b in blacks
                 if (b.start-a.start-a.width) % 360 <= 8]
        great, good = max(pairs, key=lambda ab: ab[0].width, default=(None,None))
        if great is None and len(whites)==1:
            great = whites[0]
        if good is None and len(blacks)==1:
            good=blacks[0]
        # A local angular ridge must persist along the radial segment. Broad
        # colored clothes/background can be red but do not form that ridge.
        samples = self._sample(frame,center,self._needle_radii)
        red = np.maximum(samples[:, :, 0], samples[:, :, 1])
        np.subtract(samples[:, :, 2], red, out=red)
        np.maximum(red, 0, out=red)
        # Compare angular neighbors with the same circular wrap as the profile.
        offset = 12
        contrast = np.empty_like(red)
        np.maximum(red[:-2*offset], red[2*offset:], out=contrast[offset:-offset])
        np.maximum(red[-offset:], red[offset:2*offset], out=contrast[:offset])
        np.maximum(red[-2*offset:-offset], red[:offset], out=contrast[-offset:])
        np.subtract(red, contrast, out=contrast)
        support = ((red>12)&(contrast>7)).mean(axis=1)
        profile = np.maximum(contrast,0,out=contrast).mean(axis=1)
        peaks = np.flatnonzero((profile>=np.roll(profile,1)) & (profile>np.roll(profile,-1)) &
                              (profile>10) & (support>=.55))
        candidates=[]
        for k in sorted(peaks, key=lambda i:profile[i],reverse=True):
            if any(abs((k*.5-c.angle+180)%360-180)<5 for c in candidates):
                continue
            width=1
            for direction in (-1,1):
                j=1
                while j<30 and profile[(k+direction*j)%720]>.5*profile[k]:
                    width+=1;j+=1
            width*=.5
            if width>10:
                continue
            prev,mid,nxt=profile[(k-1)%720],profile[k],profile[(k+1)%720]
            denom=2*(2*mid-prev-nxt)
            delta=float((nxt-prev)/denom) if denom>1e-6 else 0
            candidates.append(Needle(float((k+delta)*.5%360),float(red[k].mean()),
                float(support[k]),float(profile[k]),width))
            if len(candidates)==4:break
        reason='OK' if great is not None and candidates else ('NO_TARGET' if great is None else 'NO_LINE_CANDIDATE')
        return Measurement(timestamp,center,float(score),great,good,tuple(candidates),reason)


class ScaleFallbackDetector:
    """Choose a prompt scale at acquisition and keep it while tracking the ring."""

    def __init__(self, alternate_ui_scale, **detector_options):
        primary = Detector(**detector_options)
        alternate = Detector(**(detector_options | {'ui_scale': alternate_ui_scale}))
        self.roi_offset = primary.roi_offset
        self._preferred, self._alternate = primary, alternate

    def measure(self, frame, timestamp, *, center_hint=None):
        prompt = _PromptFrame(frame, timestamp)
        result = self._preferred._measure(frame, timestamp, prompt, center_hint)
        if center_hint is not None or result.reason == 'OK':
            return result
        candidate = self._alternate._measure(frame, timestamp, prompt, None)
        if candidate.reason == 'OK':
            self._preferred, self._alternate = self._alternate, self._preferred
            return candidate
        return result
