"""Forward rangefinder (hardware/ranger.py).

Off the robot: the VL53L0X binding is never imported, a fake sensor goes in
through the `_sensor` seam. Time is passed in, nothing sleeps. The pilot-side
obstacle behaviour (slow, hold, resume) is tested in the robot repo, where the
pilot lives.
"""
from hardware.ranger import Ranger
from tests.conftest import make_cfg


class FakeSensor:
    """Stands in for the VL53L0X object: get_distance() returns mm or None."""
    def __init__(self, mm=1500):
        self.mm = mm
        self.stopped = False
        self.closed = False

    def get_distance(self):
        return self.mm

    def stop_ranging(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _cfg(**over):
    base = dict(RANGE_ENABLED=True, RANGE_POLL_HZ=20, RANGE_EMA_ALPHA=1.0,   # alpha 1 = no smoothing, exact numbers
                RANGE_MAX_M=2.0, RANGE_SLOW_M=0.60, RANGE_HOLD_M=0.25, RANGE_HYST_M=0.10,
                RANGE_SLOW_SCALE=0.40, RANGE_CLEAR_S=0.75, RANGE_STALE_S=1.0)
    base.update(over)
    return make_cfg(**base)


def test_reads_metres_and_respects_poll_period():
    cfg = _cfg()
    s = FakeSensor(1234)
    r = Ranger(cfg, _sensor=s)
    assert r.distance_m() is None and not r.fresh(0.0)
    r.poll(0.0)
    assert abs(r.distance_m() - 1.234) < 1e-9 and r.fresh(0.0)
    s.mm = 500
    r.poll(0.01)                       # inside the 50 ms period: not re-read
    assert abs(r.distance_m() - 1.234) < 1e-9
    r.poll(0.06)
    assert abs(r.distance_m() - 0.5) < 1e-9


def test_invalid_read_holds_filter_and_goes_stale():
    cfg = _cfg()
    s = FakeSensor(800)
    r = Ranger(cfg, _sensor=s)
    r.poll(0.0)
    s.mm = None                        # sensor timeout / invalid status
    r.poll(0.1)
    assert abs(r.distance_m() - 0.8) < 1e-9     # held, not zeroed
    assert r.fresh(0.5)
    assert not r.fresh(1.2)                     # > RANGE_STALE_S since last valid


def test_read_that_raises_is_treated_as_invalid():
    cfg = _cfg()

    class Angry(FakeSensor):
        def get_distance(self):
            raise OSError('i2c read failed')

    r = Ranger(cfg, _sensor=Angry())
    r.poll(0.0)                        # a thrown read must not take the run down
    assert r.distance_m() is None and not r.fresh(0.0)


def test_beyond_max_clamps_to_max():
    cfg = _cfg()
    r = Ranger(cfg, _sensor=FakeSensor(3900))
    r.poll(0.0)
    assert r.distance_m() == cfg.RANGE_MAX_M


def test_ema_smooths_when_alpha_below_one():
    cfg = _cfg(RANGE_EMA_ALPHA=0.5)
    s = FakeSensor(1000)
    r = Ranger(cfg, _sensor=s)
    r.poll(0.0)
    s.mm = 500
    r.poll(0.1)
    assert abs(r.distance_m() - 0.75) < 1e-9


def test_close_stops_and_closes_sensor():
    s = FakeSensor()
    Ranger(_cfg(), _sensor=s).close()
    assert s.stopped and s.closed
