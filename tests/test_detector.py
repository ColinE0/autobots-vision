"""Classical detector on synthetic frames (vision/detector.py).

Frames are drawn at DETECT_PROC_WIDTH so no resize noise enters the asserts.
BGR colors. A LAMP is drawn at full brightness (V=255), because that is what
a lit LED does to a sensor: it clips. A SIGN is drawn at V=180: bright enough
to pass every mask floor, nowhere near clipping, which is what printed paper
does. The glow test is the only thing separating the two.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from vision.detector import Detector, make_detector
from tests.conftest import make_cfg

W, H = 320, 240
RED = (0, 0, 255)            # lit red lamp
SIGN_RED = (0, 0, 180)       # printed stop-sign red under room light
LENS_RED = (0, 0, 120)       # unlit red lens on the 3-lens module
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
MATTE_GREEN = (0, 200, 0)    # green paint: above the lamp floor, no glow
WHITE = (255, 255, 255)


def frame():
    return np.zeros((H, W, 3), np.uint8)


def labels(dets):
    return {d.label for d in dets}


def stop_sign(f, center=(160, 80), r=40):
    cv2.circle(f, center, r, SIGN_RED, -1)
    cx, cy = center
    cv2.rectangle(f, (cx - 25, cy - 7), (cx + 25, cy + 7), WHITE, -1)  # "STOP" band


def test_red_lamp_without_white_is_red_light(cfg):
    f = frame()
    cv2.circle(f, (160, 80), 40, RED, -1)
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'red_light'}
    d = dets[0]
    assert d.area_frac == pytest.approx(np.pi * 40 * 40 / (W * H), rel=0.1)


def test_red_with_white_text_is_stop_sign(cfg):
    f = frame()
    stop_sign(f)
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'stop_sign'}


def test_lit_red_lamp_with_white_core_is_a_lamp_not_a_sign(cfg):
    # The 2026-09-02 bench failure: a lit LED clips to white at its centre,
    # the white cleared STOPSIGN_WHITE_FRAC, and a red lamp was a confirmed
    # stop_sign for six seconds. Glow is judged before white.
    f = frame()
    cv2.circle(f, (160, 80), 40, RED, -1)
    cv2.circle(f, (160, 80), 15, WHITE, -1)     # ~11% of the box, over the 6% gate
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'red_light'}
    assert dets[0].area_frac == pytest.approx(np.pi * 40 * 40 / (W * H), rel=0.1)


def test_unlit_red_lens_is_nothing(cfg):
    # Dark red, no white: not a lamp (no glow) and not a sign (no text).
    f = frame()
    cv2.circle(f, (160, 80), 40, LENS_RED, -1)
    assert Detector(cfg).detect(f) == []


def test_dim_red_without_text_is_not_a_red_light(cfg):
    # SIGN_RED clears the lamp mask floor (V=180 >= 170) but does not glow.
    # The old detector called this red_light, a hold the robot cannot leave.
    f = frame()
    cv2.circle(f, (160, 80), 40, SIGN_RED, -1)
    assert Detector(cfg).detect(f) == []


def test_green_light(cfg):
    f = frame()
    cv2.circle(f, (160, 60), 30, GREEN, -1)
    assert labels(Detector(cfg).detect(f)) == {'green_light'}


def test_yellow_light(cfg):
    f = frame()
    cv2.circle(f, (160, 60), 30, YELLOW, -1)
    assert labels(Detector(cfg).detect(f)) == {'yellow_light'}


def test_matte_green_object_is_not_a_lamp(cfg):
    # Above LIGHT_V_MIN, so it is in the mask; fails the glow peak, so it is
    # a green thing in the room and not a green light.
    f = frame()
    cv2.circle(f, (160, 60), 30, MATTE_GREEN, -1)
    assert Detector(cfg).detect(f) == []


def test_one_lamp_per_frame_biggest_wins(cfg):
    # A traffic light shows one colour. The module's other lenses, or the lit
    # lamp's own halo, must not come out as a second label.
    f = frame()
    cv2.circle(f, (120, 80), 40, RED, -1)
    cv2.circle(f, (220, 80), 15, GREEN, -1)
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'red_light'}


def test_two_lamps_of_equal_size_report_no_lamp(cfg):
    # Undecidable frame: say nothing and let the TemporalFilter ride it out,
    # rather than guess green during a red.
    f = frame()
    cv2.circle(f, (100, 80), 30, RED, -1)
    cv2.circle(f, (220, 80), 30, GREEN, -1)
    assert Detector(cfg).detect(f) == []


def test_sign_and_lamp_report_together(cfg):
    f = frame()
    stop_sign(f, center=(90, 80), r=40)
    cv2.circle(f, (240, 80), 30, GREEN, -1)
    assert labels(Detector(cfg).detect(f)) == {'stop_sign', 'green_light'}


def test_bottom_band_is_ignored(cfg):
    # Red floor tape lives in the bottom DETECT_IGNORE_BOTTOM_FRAC of the frame.
    f = frame()
    cut = int(H * (1.0 - cfg.DETECT_IGNORE_BOTTOM_FRAC))
    cv2.circle(f, (160, (cut + H) // 2), 20, RED, -1)
    assert Detector(cfg).detect(f) == []


def test_tiny_blob_rejected(cfg):
    f = frame()
    cv2.circle(f, (160, 80), 3, RED, -1)     # ~0.04% of frame < 0.2% floor
    assert Detector(cfg).detect(f) == []


def test_thin_streak_rejected(cfg):
    # A 300x4 red smear: aspect 75 fails the compactness gate.
    f = frame()
    cv2.rectangle(f, (10, 60), (310, 64), RED, -1)
    assert Detector(cfg).detect(f) == []


def test_empty_frame(cfg):
    assert Detector(cfg).detect(frame()) == []


def test_large_frame_is_downscaled(cfg):
    f = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(f, (320, 160), 80, GREEN, -1)
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'green_light'}
    assert dets[0].area_frac == pytest.approx(np.pi * 80 * 80 / (640 * 480), rel=0.15)


def test_make_detector_default_classical(cfg):
    assert isinstance(make_detector(cfg), Detector)
