"""GeometricDetector on synthetic frames (vision/detector_geometric.py).

Frames are drawn at DETECT_PROC_WIDTH so no resize noise enters the asserts.
Matte props are drawn dim (V below the traffic-light brightness floor) and
lit lamps are drawn at full brightness, mirroring the physical split the
detectors rely on: signs reflect, lamps glow.
"""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from vision.detector_geometric import GeometricDetector
from tests.conftest import make_cfg

W, H = 320, 240
MATTE_RED = (0, 0, 165)     # below the light detector's V floor: reflective
GLOW_RED = (0, 0, 255)
GLOW_YELLOW = (0, 255, 255)
GLOW_GREEN = (0, 255, 0)


def frame():
    return np.zeros((H, W, 3), np.uint8)


def octagon(f, center, radius, color):
    """Filled regular octagon, flat-top orientation."""
    cx, cy = center
    pts = np.array([
        [cx + radius * math.cos(math.radians(22.5 + 45.0 * k)),
         cy + radius * math.sin(math.radians(22.5 + 45.0 * k))]
        for k in range(8)
    ], np.int32)
    cv2.fillPoly(f, [pts], color)


def labels(dets):
    return {d.label for d in dets}


@pytest.fixture
def cfg():
    return make_cfg()


def test_matte_red_octagon_is_stop_sign(cfg):
    f = frame()
    octagon(f, (160, 90), 40, MATTE_RED)
    dets = GeometricDetector(cfg).detect(f)
    assert labels(dets) == {'stop_sign'}
    # Bounding box of a flat-top octagon: 2 * R * cos(22.5 deg) per side.
    side = 2 * 40 * math.cos(math.radians(22.5))
    assert dets[0].area_frac == pytest.approx(side * side / (W * H), rel=0.2)


def test_glowing_red_circle_is_red_light(cfg):
    f = frame()
    cv2.circle(f, (160, 60), 25, GLOW_RED, -1)
    assert labels(GeometricDetector(cfg).detect(f)) == {'red_light'}


def test_yellow_light(cfg):
    f = frame()
    cv2.circle(f, (160, 60), 25, GLOW_YELLOW, -1)
    assert labels(GeometricDetector(cfg).detect(f)) == {'yellow_light'}


def test_green_light(cfg):
    f = frame()
    cv2.circle(f, (160, 60), 25, GLOW_GREEN, -1)
    assert labels(GeometricDetector(cfg).detect(f)) == {'green_light'}


def test_glowing_red_octagon_reads_as_light_not_sign(cfg):
    # Even a perfect octagon is a lamp if it glows: the stop sign detector's
    # brightness veto hands it to the traffic light detector.
    f = frame()
    octagon(f, (160, 80), 30, GLOW_RED)
    assert labels(GeometricDetector(cfg).detect(f)) == {'red_light'}


def test_light_on_same_object_suppresses_sign(cfg):
    # A lit lamp inside a red octagonal housing must read as the light, not
    # as a stop sign (the arbitration rule from the bench harness).
    f = frame()
    octagon(f, (160, 90), 40, MATTE_RED)
    cv2.circle(f, (160, 90), 12, GLOW_GREEN, -1)
    assert labels(GeometricDetector(cfg).detect(f)) == {'green_light'}


def test_bottom_band_is_ignored(cfg):
    # Red shapes at floor level are tape, not signs.
    f = frame()
    octagon(f, (160, 220), 18, MATTE_RED)
    assert GeometricDetector(cfg).detect(f) == []


def test_empty_frame(cfg):
    assert GeometricDetector(cfg).detect(frame()) == []


def test_large_frame_is_downscaled(cfg):
    f = np.zeros((480, 640, 3), np.uint8)
    octagon(f, (320, 180), 80, MATTE_RED)
    dets = GeometricDetector(cfg).detect(f)
    assert labels(dets) == {'stop_sign'}
    side = 2 * 80 * math.cos(math.radians(22.5))
    assert dets[0].area_frac == pytest.approx(side * side / (640 * 480), rel=0.2)


def test_make_detector_dispatches_geometric():
    from vision.detector import make_detector
    assert isinstance(make_detector(make_cfg(DETECTOR_BACKEND='geometric')),
                      GeometricDetector)
