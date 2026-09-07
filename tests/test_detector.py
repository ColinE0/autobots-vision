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


def lamp(f, center, r_core, r_ring, r_halo, hue=RED):
    """A lit LED as the sensor sees it: a clipped white core, a ring where
    saturation climbs back to full at V 255, a halo where V falls off to 40
    at full saturation, then a 3x3 blur. r_core close to r_ring is a blown
    lamp (close, or a long locked exposure); r_core well inside is a mild
    clip. Review 2026-09-05 harness, reproduced here so the cases it ran
    stay pinned."""
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    img = f.astype(float)
    hue = np.array(hue, float)
    img[d < r_core] = 255.0
    ring = (d >= r_core) & (d < r_ring)
    s = (d[ring] - r_core) / max(1e-6, r_ring - r_core)
    img[ring] = (1 - s)[:, None] * 255.0 + s[:, None] * hue
    halo = (d >= r_ring) & (d < r_halo)
    t = (d[halo] - r_ring) / max(1e-6, r_halo - r_ring)
    v = 255.0 * (1 - t) + 40.0 * t
    img[halo] = (v / 255.0)[:, None] * hue
    out = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    f[:] = out
    return f


def octagon_sign(f, center=(160, 80), r=40, v=180, letters=True, border=True):
    """A printed stop sign as a regular octagon at brightness v, with the
    white STOP text and white border a real sign carries."""
    cx, cy = center
    pts = np.array([[cx + r * np.cos(np.pi / 8 + k * np.pi / 4),
                     cy + r * np.sin(np.pi / 8 + k * np.pi / 4)]
                    for k in range(8)], np.int32)
    cv2.fillPoly(f, [pts], (0, 0, v))
    if border:
        cv2.polylines(f, [pts], True, WHITE, 2)
    if letters:
        cv2.putText(f, 'STOP', (cx - 30, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, WHITE, 2)
    return f


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


# Blown-out lamps (review 2026-09-05). A lit LED that clips almost to its rim
# loses its colour: the saturation floor deletes the white core, the ring and
# halo that survive fail the glow mean, and the white core then read as STOP
# text. Every case below was a confirmed stop_sign before the clipped-core
# test, which on the robot ends the run at a red light.

@pytest.mark.parametrize('name,core,ring,halo', [
    ('blown out', 29, 31, 45),
    ('blown with wide halo', 29, 31, 60),
    ('fully white', 31, 32, 50),
    ('close and blown', 68, 70, 95),
    ('far and blown', 11, 12, 18),
])
def test_blown_out_red_lamp_is_red_light(cfg, name, core, ring, halo):
    f = lamp(frame(), (160, 80), core, ring, halo)
    dets = Detector(cfg).detect(f)
    assert labels(dets) == {'red_light'}, name


def test_blown_out_yellow_and_green_lamps_keep_their_colour(cfg):
    for hue, label in ((YELLOW, 'yellow_light'), (GREEN, 'green_light')):
        f = lamp(frame(), (160, 80), 29, 31, 45, hue=hue)
        assert labels(Detector(cfg).detect(f)) == {label}


def test_heavily_clipped_lamp_still_passes_on_glow_alone(cfg, monkeypatch):
    # The core test is off the hot path: a lamp that still glows never
    # reaches it. Guard that, so the extra connected-components pass cannot
    # creep into every frame.
    import vision.detector as det_mod

    def never(*a, **k):
        raise AssertionError('clipped-core test ran on a glowing lamp')
    monkeypatch.setattr(det_mod, '_clipped_core', never)
    f = lamp(frame(), (160, 80), 24, 30, 40)
    assert labels(Detector(cfg).detect(f)) == {'red_light'}


def test_white_disc_on_dim_red_is_a_lamp_not_a_sign(cfg):
    # Worst case for the core test: printed-sign red with a round white
    # centre. A round central white mass is a lamp signature and a real
    # sign never has one; if this is ever wrong it costs a hold, not a run.
    f = frame()
    cv2.circle(f, (160, 80), 40, SIGN_RED, -1)
    cv2.circle(f, (160, 80), 20, WHITE, -1)
    assert labels(Detector(cfg).detect(f)) == {'red_light'}


@pytest.mark.parametrize('v', [120, 180, 230])
def test_octagon_sign_with_text_and_border_is_stop_sign(cfg, v):
    # Shadow, room light, and a brightly lit sign: the border ring is hollow
    # and the text is a wide band, so neither reads as a core.
    f = octagon_sign(frame(), v=v)
    assert labels(Detector(cfg).detect(f)) == {'stop_sign'}


def test_octagon_sign_with_text_only_is_stop_sign(cfg):
    f = octagon_sign(frame(), border=False)
    assert labels(Detector(cfg).detect(f)) == {'stop_sign'}


def test_red_octagon_without_white_is_nothing(cfg):
    # A red poster: no glow, no core, no STOP text. Not a sign, not a lamp.
    f = octagon_sign(frame(), v=200, letters=False, border=False)
    assert Detector(cfg).detect(f) == []


# Lighting (review 2026-09-07). A printed sign is a reflector: a darker
# exposure dims its letters and its red together, so white is judged
# relative to the blob's own red, and inside the blob's filled contour.

def dim(f, k):
    """The same scene at k times the exposure, clipped like a sensor."""
    return np.clip(f.astype(np.float32) * k, 0, 255).astype(np.uint8)


@pytest.mark.parametrize('k', [0.85, 0.7, 0.5])
def test_sign_survives_a_darker_exposure(cfg, k):
    # At 0.7x the letters sat under the old fixed V 170 floor and the sign
    # vanished, which on the robot is a run that never finishes.
    f = dim(octagon_sign(frame(), v=200), k)
    assert labels(Detector(cfg).detect(f)) == {'stop_sign'}, k


def test_white_floor_follows_the_blob(cfg):
    # Letters only 5% brighter than the red they sit on are not white under
    # the shipped ratio (1.1) and the sign is nothing; lowering the ratio
    # under that makes them white again. Pins that the floor is the blob's
    # own red times STOPSIGN_WHITE_REL and nothing else.
    f = octagon_sign(frame(), v=200, letters=False, border=False)
    grey = (210, 210, 210)                       # V 210 on red V 200
    cv2.putText(f, 'STOP', (130, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, grey, 2)
    assert Detector(cfg).detect(f) == []
    cfg.STOPSIGN_WHITE_REL = 1.0
    assert labels(Detector(cfg).detect(f)) == {'stop_sign'}


def test_blown_lamp_on_a_white_wall_is_a_lamp_not_a_sign(cfg):
    # The lamp's box has wall in its corners. Counted against the box, that
    # wall read as STOP text and a red light ended the run; white is now
    # judged inside the blob's own outline.
    f = np.full((H, W, 3), 255, np.uint8)
    lamp(f, (160, 80), 29, 31, 45)
    assert labels(Detector(cfg).detect(f)) == {'red_light'}


def test_orange_yellow_lamp_is_yellow_not_nothing(cfg):
    # Hue 13 on OpenCV's scale: an amber lamp under a warm white-balance
    # lock. The old bands left hue 11 to 17 unclaimed, and an unclaimed
    # yellow is driven through.
    f = lamp(frame(), (160, 80), 20, 24, 34, hue=(0, 110, 255))
    assert labels(Detector(cfg).detect(f)) == {'yellow_light'}


def test_bright_flat_sign_reads_as_a_lamp_until_the_spread_veto_is_on(cfg):
    # A printed sign lit to V 240 clears every glow floor. With the veto off
    # (shipped) it is a red_light, a hold that never clears; with it on, the
    # flat brightness profile says not a lamp and the text says sign.
    f = octagon_sign(frame(), v=240)
    assert labels(Detector(cfg).detect(f)) == {'red_light'}
    cfg.LAMP_MIN_V_SPREAD = 30
    assert labels(Detector(cfg).detect(f)) == {'stop_sign'}


def test_spread_veto_keeps_every_rendered_lamp(cfg):
    # Rendered lamps spread 61 to 75 between their 10th and 90th percentile;
    # the veto at 30 must not touch them.
    cfg.LAMP_MIN_V_SPREAD = 30
    for core, ring, halo in ((29, 31, 45), (24, 30, 40), (0, 20, 30), (68, 70, 95)):
        f = lamp(frame(), (160, 80), core, ring, halo)
        assert labels(Detector(cfg).detect(f)) == {'red_light'}, (core, ring, halo)


def test_core_test_is_config_driven(cfg):
    # Turning the core gate impossible restores the pre-fix reading, which
    # pins that the fix lives in the four LAMP_CORE_* knobs and nowhere else.
    cfg.LAMP_CORE_MIN_FRAC = 2.0
    f = lamp(frame(), (160, 80), 29, 31, 45)
    assert labels(Detector(cfg).detect(f)) == {'stop_sign'}
