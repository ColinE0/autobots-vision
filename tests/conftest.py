"""
Shared test plumbing. Everything runs off the robot: no I2C, no camera, no Pi.
Sensors are faked through the seam each driver exposes, and time is passed in
explicitly, so nothing sleeps.
"""
import copy
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest


def make_cfg(**overrides):
    """Deep-copied snapshot of config.py the test can mutate freely."""
    import config
    ns = types.SimpleNamespace()
    for k in dir(config):
        if k.isupper():
            setattr(ns, k, copy.deepcopy(getattr(config, k)))
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def cfg():
    return make_cfg()
