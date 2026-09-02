"""Classical detector on synthetic frames (vision/detector.py).

Frames are drawn at DETECT_PROC_WIDTH so no resize noise enters the asserts.
BGR colors: red (0,0,255), yellow (0,255,255), green (0,255,0).
"""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from vision.detector import Detector, make_detector
from tests.conftest import make_cfg

W, H = 320, 240
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def frame():
    return np.zeros((H, W, 3), np.uint8)


def labels(dets):
    return {d.label for d in dets}


def test_red_lamp_without_white_is_red_light(cfg):
    f = frame()
    cv2.circle(f, (160, 80), 40, RED, -1)
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'red_light'}
    d = dets[0]
    assert d.area_frac == pytest.approx(np.pi * 40 * 40 / (W * H), rel=0.1)


def test_red_with_white_text_is_stop_sign(cfg):
    f = frame()
    cv2.circle(f, (160, 80), 40, RED, -1)
    cv2.rectangle(f, (135, 73), (185, 87), WHITE, -1)   # the "STOP" band
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'stop_sign'}


def test_green_light(cfg):
    f = frame()
    cv2.circle(f, (160, 60), 30, GREEN, -1)
    assert labels(Detector(cfg).detect(f)) == {'green_light'}


def test_yellow_light(cfg):
    f = frame()
    cv2.circle(f, (160, 60), 30, YELLOW, -1)
    assert labels(Detector(cfg).detect(f)) == {'yellow_light'}


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
