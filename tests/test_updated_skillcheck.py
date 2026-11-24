"""The 2026-10-24 user screenshot is CV evidence, not a game hit verdict."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from vd.engine import Engine
from vd.image_io import read_image
from vd.vision import Detector, ScaleFallbackDetector, retained_target


FIXTURE = Path(__file__).parent / 'fixtures' / 'skillcheck_20261024.png'
WINDOWS_768_UI_SCALE = 768 / 1080


def updated_frame():
    frame = read_image(FIXTURE, cv2.IMREAD_COLOR)
    assert frame is not None
    return frame


def windows_768_roi(ui_scale, offset):
    screen_width, screen_height = 1366, 768
    roi_width, roi_height = 320, 240
    left = (screen_width - roi_width) // 2
    top = (screen_height - roi_height) // 2
    center_x = screen_width / 2 + offset[0] * ui_scale
    center_y = screen_height / 2 + offset[1] * ui_scale
    transform = np.array([[ui_scale, 0, center_x - 122 * ui_scale],
                          [0, ui_scale, center_y - 106.5 * ui_scale]], np.float32)
    screen = cv2.warpAffine(updated_frame(), transform, (screen_width, screen_height),
                            borderValue=(18, 21, 25))
    return screen[top:top + roi_height, left:left + roi_width]


def moving_updated_frame(angle, great_start=153):
    """Synthetic ring/motion with the screenshot's actual prompt and red sprite."""
    source = updated_frame()
    frame = np.full_like(source, (18, 21, 25))
    cv2.circle(frame, (122, 106), 67, (220, 225, 229), 2, cv2.LINE_AA)
    for start, width, thickness, color in (
            (great_start, 10, 10, (255, 255, 255)),
            (great_start+11, 42, 7, (8, 9, 10))):
        radians = np.deg2rad(np.linspace(start, start+width, round(width*2)+1))
        outer = np.column_stack((122+(66.5+thickness/2)*np.cos(radians),
                                 106.5+(66.5+thickness/2)*np.sin(radians)))
        inner = np.column_stack((122+(66.5-thickness/2)*np.cos(radians),
                                 106.5+(66.5-thickness/2)*np.sin(radians)))
        polygon = np.rint(np.concatenate((outer, inner[::-1]))).astype(np.int32)
        cv2.fillPoly(frame, [polygon], color, lineType=cv2.LINE_AA)
    frame[89:124, 87:157] = source[89:124, 87:157]

    other = np.maximum(source[:, :, 0], source[:, :, 1])
    red = source[:, :, 2].astype(float) - other
    y, x = np.indices(red.shape)
    radius = np.hypot(x-122, y-106.5)
    # Exclude the pink-tinted white target from the extracted moving sprite.
    mask = ((red > 40) & (other < 130) & (radius > 25) & (radius < 95))
    transform = cv2.getRotationMatrix2D((122, 106.5), -(angle-152.12), 1)
    arrow = cv2.warpAffine(source, transform, (246, 201))
    alpha = cv2.warpAffine(mask.astype(np.uint8)*255, transform, (246, 201))
    alpha = alpha[:, :, None] / 255
    return (frame*(1-alpha) + arrow*alpha).astype(np.uint8)


def test_updated_prompt_is_acquired_without_a_previous_center():
    measured = Detector().measure(updated_frame(), 1.0)

    assert measured.reason == 'OK'
    assert measured.prompt_score >= .80
    # Hand checked from the screenshot: the ring is centered near (122, 106),
    # the white sector is near 153..163 degrees and the red arrow near 152.
    assert measured.center == pytest.approx((122, 106.5), abs=1)
    assert measured.great is not None
    assert measured.great.start == pytest.approx(153.4, abs=2)
    assert measured.great.width == pytest.approx(10.2, abs=2)
    assert measured.good is not None
    assert measured.good.width == pytest.approx(42.8, abs=3)
    assert len(measured.candidates) == 1
    assert measured.candidates[0].angle == pytest.approx(152.1, abs=2)


def test_custom_prompt_template_supports_unicode_path(tmp_path):
    template = tmp_path / 'шаблон.png'
    source = Path(__file__).parents[1] / 'assets' / 'space_template_20261024.png'
    template.write_bytes(source.read_bytes())

    assert Detector(template=template).measure(updated_frame(), 1.0).reason == 'OK'


@pytest.mark.parametrize('scale', (1.0, 1.2))
@pytest.mark.parametrize('center', ((160, 162.5), (170, 82.5)))
def test_updated_prompt_works_in_live_compact_roi(center, scale):
    source = updated_frame()
    transform = np.array([[scale, 0, center[0] - 122*scale],
                          [0, scale, center[1] - 106.5*scale]], np.float32)
    frame = cv2.warpAffine(source, transform, (320, 260))
    compact = cv2.copyMakeBorder(frame, 2, 0, 0, 0, cv2.BORDER_CONSTANT)[:250, 75:255]

    measured = Detector(roi_offset=(75, -2), ui_scale=scale).measure(compact, 1.0)

    assert measured.reason == 'OK'
    assert measured.center == pytest.approx((center[0]-75, center[1]+2), abs=1)
    assert measured.great is not None
    assert measured.great.center == pytest.approx(158.5, abs=2)
    assert len(measured.candidates) == 1
    assert measured.candidates[0].angle == pytest.approx(152.1, abs=2)


@pytest.mark.parametrize('ui_scale', (1.0, WINDOWS_768_UI_SCALE))
@pytest.mark.parametrize('offset', ((0, 42.5), (10, -37.5)))
def test_windows_768_capture_fixture_accepts_both_ui_scales(ui_scale, offset):
    measured = ScaleFallbackDetector(WINDOWS_768_UI_SCALE).measure(
        windows_768_roi(ui_scale, offset), 1.0)

    assert measured.reason == 'OK'
    assert measured.prompt_score >= .80
    assert measured.center == pytest.approx((160 + offset[0] * ui_scale,
                                              120 + offset[1] * ui_scale), abs=1)
    assert measured.great is not None
    assert measured.great.start == pytest.approx(153.4, abs=2)
    assert measured.great.width == pytest.approx(10.2, abs=2)
    assert len(measured.candidates) == 1
    assert measured.candidates[0].angle == pytest.approx(152.1, abs=2)


def test_windows_768_fixture_changes_scale_only_on_acquisition():
    detector = ScaleFallbackDetector(WINDOWS_768_UI_SCALE)
    full_size = windows_768_roi(1.0, (0, 42.5))
    fractional_size = windows_768_roi(WINDOWS_768_UI_SCALE, (0, 42.5))
    first = detector.measure(fractional_size, 1.0)
    assert first.reason == 'OK'

    tracking = detector.measure(full_size, 1.02, center_hint=first.center)
    reacquired = detector.measure(full_size, 1.04)

    assert tracking.prompt_score < .80
    assert tracking.great is None
    assert reacquired.reason == 'OK'
    assert reacquired.center == pytest.approx((160, 162.5), abs=1)


def test_updated_prompt_survives_video_compression_and_brightness_change():
    frame = cv2.convertScaleAbs(updated_frame(), alpha=.85, beta=12)
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    measured = Detector().measure(decoded, 1.0)

    assert measured.prompt_score >= .80
    assert measured.reason == 'OK'
    assert measured.great is not None
    assert measured.candidates[0].angle == pytest.approx(152.1, abs=2)


def test_updated_prompt_confirms_engine_geometry_instead_of_waiting_forever():
    detector = Detector()
    engine = Engine()
    for at in (1.0, 1.02):
        engine.observe(detector.measure(updated_frame(), at), now=at)

    assert engine.active
    assert engine.generation == 1
    assert engine.target is not None
    assert engine.target.center == pytest.approx(158.5, abs=2)
    assert [event['kind'] for event in engine.take_events()] == ['BEGIN']


def test_screenshot_without_the_prompt_does_not_acquire_from_ring_alone():
    frame = updated_frame()
    frame[87:126, 85:160] = 0

    measured = Detector().measure(frame, 1.0)

    assert measured.reason == 'NO_PROMPT'
    assert measured.prompt_score < .80


@pytest.mark.parametrize('angle', (156, 158.5, 160))
def test_new_arrow_occlusion_retains_previously_observed_white_sector(angle):
    detector = Detector()
    first = detector.measure(moving_updated_frame(140), 1.0)
    assert first.reason == 'OK'
    measured = detector.measure(moving_updated_frame(angle), 1.02,
                                center_hint=first.center)

    tracked = retained_target(measured, great=first.great, good=first.good,
                              center=first.center)

    assert tracked.reason == 'OK'
    assert tracked.great_source == 'TRACKED_NEEDLE_OCCLUSION'
    assert tracked.great == first.great
    assert len(tracked.candidates) == 1
    assert tracked.candidates[0].angle == pytest.approx(angle, abs=1)


def test_new_arrow_does_not_hide_a_real_target_relocation():
    detector = Detector()
    first = detector.measure(moving_updated_frame(140), 1.0)
    measured = detector.measure(moving_updated_frame(158.5, great_start=30), 1.02,
                                center_hint=first.center)

    tracked = retained_target(measured, great=first.great, good=first.good,
                              center=first.center)

    assert tracked.great is not None
    assert tracked.great_source == 'MEASURED'
    assert tracked.great.center == pytest.approx(35, abs=2)


def test_new_prompt_and_arrow_drive_one_great_claim_with_fake_motion():
    detector = Detector()
    engine = Engine(lead_seconds=.004, lead_uncertainty=.020)
    events = []
    for index in range(72):
        at = index/120
        measured = detector.measure(moving_updated_frame(90+300*at), at,
                                    center_hint=engine.center)
        measured = retained_target(measured, great=engine.target, good=engine.good,
                                    center=engine.center)
        engine.observe(measured, now=at)
        plan = engine.planner.current
        wake = plan.press_at if plan is not None and at <= plan.press_at < at+1/120 else at
        engine.poll(wake)
        events.extend(engine.take_events())

    claims = [event for event in events if event['kind'] == 'PRESS_CLAIM']
    assert len(claims) == 1
    assert claims[0]['plan']['target_grade'] == 'GREAT'
    # This checks the synthetic trajectory only, not an actual game outcome.
    impact = 90+300*(claims[0]['at']+.004)
    assert 153 <= impact <= 163
