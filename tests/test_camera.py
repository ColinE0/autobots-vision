"""Camera backends (hardware/camera.py): factory dispatch and the CSI path.

The picamera2 object is faked, so the suite still runs off-Pi. The fake
serves a fixed number of frames, then parks the capture thread on an event
until close(), the same shape as real hardware blocking for the next frame.
"""
import threading
import time

import numpy as np
import pytest

import hardware.camera as camera_mod
from hardware.camera import CsiCamera, make_camera
from tests.conftest import make_cfg


class FakePicamera2:
    """Minimal stand-in for picamera2.Picamera2."""

    def __init__(self, frames=1, first_frame_none=False):
        self.video_config = None
        self.started = False
        self.closed = False
        self.controls_set = []
        self.metadata = {'ColourGains': (1.8, 1.5)}
        self._frames_left = frames
        self._first_none = first_frame_none
        self._stopped = threading.Event()

    def create_video_configuration(self, main=None, controls=None):
        return {'main': main, 'controls': controls}

    def configure(self, cfg):
        self.video_config = cfg

    def start(self):
        self.started = True

    def capture_array(self, name='main'):
        if self._first_none:
            self._first_none = False
            return None
        if self._frames_left > 0:
            self._frames_left -= 1
            return np.zeros((240, 320, 3), np.uint8)
        # out of frames: block like real hardware until stop()
        self._stopped.wait(timeout=5.0)
        raise RuntimeError('camera stopped')

    def capture_metadata(self):
        return dict(self.metadata)

    def set_controls(self, controls):
        self.controls_set.append(dict(controls))

    def stop(self):
        self._stopped.set()

    def close(self):
        self.closed = True


def test_make_camera_dispatch(monkeypatch):
    built = []
    monkeypatch.setattr(camera_mod, 'CsiCamera', lambda cfg: built.append('csi'))
    monkeypatch.setattr(camera_mod, 'UsbCamera', lambda cfg: built.append('usb'))
    make_camera(make_cfg(CAMERA_BACKEND='csi'))
    make_camera(make_cfg(CAMERA_BACKEND='usb'))
    assert built == ['csi', 'usb']


def test_make_camera_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_camera(make_cfg(CAMERA_BACKEND='firewire'))


def test_csi_configures_stream_from_config():
    cfg = make_cfg(CAMERA_WIDTH=320, CAMERA_HEIGHT=240, CAMERA_FPS=30)
    fake = FakePicamera2()
    cam = CsiCamera(cfg, _picam2=fake)
    try:
        assert fake.started
        # RGB888 is B,G,R in memory (libcamera names run backwards), which is
        # OpenCV's order; the detector's HSV math depends on this exact string.
        assert fake.video_config['main'] == {'size': (320, 240), 'format': 'RGB888'}
        assert fake.video_config['controls'] == {'FrameRate': 30.0}
    finally:
        cam.close()
    assert fake.closed


def test_csi_read_returns_first_frame_and_close_joins():
    cam = CsiCamera(make_cfg(), _picam2=FakePicamera2(frames=1))
    fid, frame = cam.read()
    assert fid == 1
    assert frame.shape == (240, 320, 3)
    cam.close()
    assert not cam._thread.is_alive()


def test_csi_thread_publishes_newer_frames():
    cam = CsiCamera(make_cfg(), _picam2=FakePicamera2(frames=3))
    try:
        deadline = time.monotonic() + 2.0
        fid = 0
        while time.monotonic() < deadline:
            fid, _ = cam.read()
            if fid >= 3:
                break
            time.sleep(0.005)
        assert fid == 3          # init frame plus two more from the thread
    finally:
        cam.close()


def test_csi_awb_lock_freezes_measured_gains(monkeypatch):
    monkeypatch.setattr(camera_mod, '_AWB_WARMUP_S', 0.0)
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_LOCK_AWB=True), _picam2=fake)
    try:
        assert {'AwbEnable': False, 'ColourGains': (1.8, 1.5)} in fake.controls_set
    finally:
        cam.close()


def test_csi_first_capture_failure_raises():
    with pytest.raises(RuntimeError):
        CsiCamera(make_cfg(), _picam2=FakePicamera2(first_frame_none=True))
