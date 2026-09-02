"""TemporalFilter K-of-N confirmation (vision/detector.py)."""
from vision.detector import Detection, TemporalFilter
from tests.conftest import make_cfg


def _f(cfg=None):
    return TemporalFilter(cfg or make_cfg())   # N=3, K=2 from config


def test_single_frame_not_confirmed():
    t = _f()
    t.update([Detection('red_light', 0.01)])
    assert not t.confirmed('red_light')


def test_two_of_three_confirmed():
    t = _f()
    t.update([Detection('red_light', 0.01)])
    t.update([Detection('red_light', 0.012)])
    assert t.confirmed('red_light')


def test_glint_then_gone_never_confirms():
    t = _f()
    t.update([Detection('stop_sign', 0.02)])
    t.update([])
    t.update([])
    assert not t.confirmed('stop_sign')


def test_confirmation_decays_as_window_slides():
    t = _f()
    t.update([Detection('red_light', 0.01)])
    t.update([Detection('red_light', 0.01)])
    assert t.confirmed('red_light')
    t.update([])
    assert t.confirmed('red_light')      # still 2 of last 3
    t.update([])
    assert not t.confirmed('red_light')  # now 1 of last 3


def test_area_returns_most_recent():
    t = _f()
    t.update([Detection('stop_sign', 0.010)])
    t.update([Detection('stop_sign', 0.020)])
    assert t.area('stop_sign') == 0.020
    t.update([])
    assert t.area('stop_sign') == 0.020  # last seen inside the window
    t.update([])
    t.update([])
    assert t.area('stop_sign') == 0.0    # slid out


def test_labels_tracked_independently():
    t = _f()
    t.update([Detection('red_light', 0.01), Detection('green_light', 0.01)])
    t.update([Detection('green_light', 0.01)])
    assert t.confirmed('green_light')
    assert not t.confirmed('red_light')


def test_reset_clears_window():
    t = _f()
    t.update([Detection('red_light', 0.01)])
    t.update([Detection('red_light', 0.01)])
    t.reset()
    assert not t.confirmed('red_light')
    assert t.area('red_light') == 0.0
