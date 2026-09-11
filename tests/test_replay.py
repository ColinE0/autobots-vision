"""Replay accounting (tools/replay.py).

The detector is stubbed: what is under test is the bookkeeping a scrolling
console cannot do, above all the flap count, which is the number that tells a
real detection apart from a threshold sitting on the edge of the signal.
"""
import numpy as np

from tests.conftest import make_cfg
from tools.replay import analyse


class Det:
    """Detector stub: replays a scripted list of per-frame label lists."""

    def __init__(self, script):
        self.script = list(script)
        self.i = -1

    def detect(self, frame):
        self.i += 1
        labels = self.script[self.i] if self.i < len(self.script) else []
        return [Detection(lbl, area) for lbl, area in labels]


class Detection:
    def __init__(self, label, area_frac):
        self.label = label
        self.area_frac = area_frac


def frames(n):
    return [np.zeros((240, 320, 3), np.uint8) for _ in range(n)]


def test_raw_counts_frames_areas_and_flaps():
    script = [[('red_light', 0.01)], [], [('red_light', 0.03)], []]
    stats = analyse(frames(4), make_cfg(), raw=True, det=Det(script))
    red = stats['red_light']
    assert stats['frames'] == 4
    assert red['frames'] == 2
    assert red['areas'] == [0.01, 0.03]
    # on, off, on, off is two flaps: four edges.
    assert red['transitions'] == 4
    assert red['longest'] == 1


def test_longest_streak_is_the_uninterrupted_run():
    script = [[('green_light', 0.02)]] * 5 + [[]] + [[('green_light', 0.02)]] * 2
    stats = analyse(frames(8), make_cfg(), raw=True, det=Det(script))
    assert stats['green_light']['longest'] == 5
    assert stats['green_light']['frames'] == 7


def test_a_label_never_seen_reports_nothing():
    stats = analyse(frames(3), make_cfg(), raw=True, det=Det([[], [], []]))
    for lbl in ('stop_sign', 'red_light', 'yellow_light', 'green_light'):
        assert stats[lbl]['frames'] == 0
        assert stats[lbl]['transitions'] == 0


def test_confirmed_view_absorbs_a_single_frame_flip():
    # The whole point of the K-of-N filter: one frame of noise inside a run
    # must not register as a detection, so the confirmed view sees fewer
    # flaps than the raw view over the same script.
    script = ([[('red_light', 0.02)]] * 6 + [[]] + [[('red_light', 0.02)]] * 6)
    cfg = make_cfg()
    raw = analyse(frames(13), cfg, raw=True, det=Det(script))
    confirmed = analyse(frames(13), cfg, raw=False, det=Det(script))
    assert raw['red_light']['transitions'] > confirmed['red_light']['transitions']
