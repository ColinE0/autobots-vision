"""
The HSV probe reports the detector's own numbers and names the first gate a
blob fails. Synthetic frames, same conventions as test_detector.py: a lamp
clips to V 255, a lens or a sign sits well under.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from vision.detector import Detector, _glows
from vision.probe import Probe, centre_hsv, glow_stats, format_line
from tests.conftest import make_cfg

W, H = 320, 240
RED = (0, 0, 255)
GREEN = (0, 255, 0)
LENS_RED = (0, 0, 120)          # unlit red lens, V 120


@pytest.fixture
def cfg():
    return make_cfg()


def frame():
    return np.zeros((H, W, 3), np.uint8)


def lamp(f, center, hue=RED, r_core=12, r_ring=18, r_halo=28):
    b, g, r = hue
    cv2.circle(f, center, r_halo, (b // 2, g // 2, r // 2), -1)
    cv2.circle(f, center, r_ring, hue, -1)
    cv2.circle(f, center, r_core, (255, 255, 255), -1)
    return f


def test_centre_patch_reads_the_prop_colour():
    f = frame()
    cv2.rectangle(f, (140, 100), (180, 140), (0, 0, 200), -1)   # pure red, V 200
    h, s, v = centre_hsv(f)
    assert h == 0 and s == 255 and v == 200


def test_glow_stats_agrees_with_the_detector_gate(cfg):
    gate = tuple(cfg.LAMP_GLOW['red'])
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 400))
        vals = rng.integers(0, 256, n).astype(np.uint8)
        st = glow_stats(vals, cfg.LIGHT_V_MIN, gate,
                        cfg.LAMP_MIN_V_SPREAD, cfg.LAMP_MIN_BRIGHT_PIXELS)
        assert (st['fail'] is None) == _glows(vals, cfg.LIGHT_V_MIN, gate,
                                              cfg.LAMP_MIN_V_SPREAD,
                                              cfg.LAMP_MIN_BRIGHT_PIXELS)


def test_glow_stats_names_the_first_failing_gate(cfg):
    gate = (220, 240, 0.40, 220)
    assert glow_stats(np.array([], np.uint8), 170, gate)['fail'] == 'no pixels'
    dim = np.full(100, 150, np.uint8)
    assert glow_stats(dim, 170, gate)['fail'].startswith('bright pixels 0')
    # 20 bright pixels of 100: count passes, share 0.20 < 0.40
    mixed = np.array([250] * 20 + [100] * 80, np.uint8)
    assert glow_stats(mixed, 170, gate)['fail'].startswith('bright share 0.20')
    lit = np.full(100, 250, np.uint8)
    st = glow_stats(lit, 170, gate)
    assert st['fail'] is None and st['bright'] == 100 and st['mean'] == 250.0


def test_lit_lamp_glows_and_unlit_lens_fails_with_a_reason(cfg):
    p = Probe(cfg)
    f = lamp(frame(), (160, 80), hue=GREEN)
    st = p.measure(f)['green']
    assert st is not None and st['fail'] is None
    assert st['area_frac'] > cfg.DETECT_MIN_AREA_FRAC
    f = frame()
    cv2.circle(f, (160, 80), 30, LENS_RED, -1)
    st = p.measure(f)['red']
    assert st is not None and st['fail'] is not None
    assert st['bright'] == 0
    assert 110 <= st['v_mean'] <= 125


def test_probe_sees_what_the_detector_sees(cfg):
    # A blob the probe calls GLOWS is one the detector labels; a blob it
    # fails is not a lamp to the detector either.
    p, d = Probe(cfg), Detector(cfg)
    f = lamp(frame(), (160, 80), hue=RED)
    st = p.measure(f)['red']
    # Red masks at SIGN_V_MIN so the halo dilutes the bright share; the
    # detector takes this lamp on its clipped core, and the probe says so.
    assert st['fail'] is None or st['core']
    assert {x.label for x in d.detect(f)} == {'red_light'}
    f = frame()
    cv2.circle(f, (160, 80), 30, LENS_RED, -1)
    st = p.measure(f)['red']
    assert st['fail'] is not None and not st['core'] and st['core_px'] == 0
    assert d.detect(f) == []


def test_blown_out_lamp_reports_core(cfg):
    f = lamp(frame(), (160, 80), hue=RED)
    line = format_line(centre_hsv(f), Probe(cfg).measure(f))
    assert '[GLOWS]' in line or '[CORE (' in line


def test_nothing_in_frame_prints_dashes(cfg):
    blobs = Probe(cfg).measure(frame())
    assert all(v is None for v in blobs.values())
    line = format_line((0, 0, 0), blobs)
    assert line.endswith('R: - | Y: - | G: -')


def test_format_line_carries_the_verdict(cfg):
    f = lamp(frame(), (160, 80), hue=GREEN)
    line = format_line(centre_hsv(f), Probe(cfg).measure(f))
    assert '[GLOWS]' in line and 'G: ' in line
    f = frame()
    cv2.circle(f, (160, 80), 30, LENS_RED, -1)
    line = format_line(centre_hsv(f), Probe(cfg).measure(f))
    assert 'bright pixels 0 < ' in line
