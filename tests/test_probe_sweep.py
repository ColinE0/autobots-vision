"""Exposure sweep accounting (tools/probe_hsv.py).

The camera is faked and the real Probe and detector run on synthetic frames.
What is under test is the tabulation: one row per exposure per colour, rates
expressed against the frames actually captured at that step, and the most
common failure carried through rather than the last one.
"""
import cv2
import numpy as np
import pytest

import tools.probe_hsv as probe_hsv


class Log:
    def __init__(self):
        self.lines = []

    def line(self, text, echo=True):
        self.lines.append(text)
        return text

    def close(self):
        pass


def lamp_frame():
    f = np.full((240, 320, 3), 15, np.uint8)
    cv2.circle(f, (160, 90), 30, (0, 0, 255), -1)     # saturated red lamp
    cv2.circle(f, (160, 90), 11, (200, 200, 255), -1)  # clipped white core
    return f


class FakeCam:
    def __init__(self, frame):
        self._frame = frame
        self.i = 0
        self.closed = False

    def read(self):
        self.i += 1
        return self.i, self._frame

    def close(self):
        self.closed = True


@pytest.fixture
def fake_camera(monkeypatch):
    built = []

    def make(cfg):
        # Record the exposure each step asked for: a sweep that does not
        # actually change exposure between steps is worthless.
        built.append(cfg.CAMERA_EXPOSURE_US)
        cam = FakeCam(lamp_frame())
        built.append(cam)
        return cam

    monkeypatch.setattr(probe_hsv, 'make_camera', make)
    return built


def test_sweep_walks_every_exposure_and_reopens_the_camera(fake_camera):
    rows, _ = probe_hsv.sweep([8000, 4000], 0.05, Log())
    exposures = [x for x in fake_camera if isinstance(x, int)]
    cams = [x for x in fake_camera if isinstance(x, FakeCam)]
    assert exposures == [8000, 4000]
    assert all(c.closed for c in cams), 'each step must close its camera'
    assert {r['us'] for r in rows} == {8000, 4000}


def test_sweep_reports_the_red_lamp_it_was_shown(fake_camera):
    rows, _ = probe_hsv.sweep([4000], 0.05, Log())
    red = [r for r in rows if r['colour'] == 'red']
    assert len(red) == 1
    r = red[0]
    assert r['frames'] > 0
    assert r['blob'] == r['frames']          # the lamp is in every frame
    assert r['area'] > 0
    assert r['core'] > 0                     # the clipped core is measured


def test_sweep_measures_the_scene_even_with_no_coloured_blob(fake_camera):
    # A dark frame yields no blob. The scene numbers are then the ONLY
    # evidence, and without them an empty table cannot be read.
    rows, scenes = probe_hsv.sweep([4000], 0.05, Log())
    assert [s['us'] for s in scenes] == [4000]
    assert scenes[0]['peak_v'] is not None


def test_report_names_which_empty_result_it_is():
    # "Nothing detected" must distinguish an unlit scene from a lamp the
    # exposures all missed, or a venue trip produces an unreadable table.
    log = Log()
    probe_hsv.report_sweep([], [{'us': 4000, 'frames': 100,
                                 'centre_v': 3, 'peak_v': 9}], log=log)
    blob = ' '.join(log.lines)
    assert 'no red, yellow or green blob' in blob
    assert 'nothing lit' in blob


def test_report_survives_a_sweep_with_no_data_at_all():
    log = Log()
    probe_hsv.report_sweep([], (), log=log)
    assert any('exposure' in line for line in log.lines)
