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

    def __init__(self, frames=1, first_frame_none=False, frame_delay=0.0):
        self.video_config = None
        self.started = False
        self.closed = False
        self.controls_set = []
        self.metadata = {'ColourGains': (1.8, 1.5),
                         'ExposureTime': 19999, 'AnalogueGain': 2.5}
        self._frames_left = frames
        self._first_none = first_frame_none
        self._frame_delay = frame_delay     # per-frame pacing, like a real sensor
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
            if self._frame_delay:
                time.sleep(self._frame_delay)
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
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_LOCK_AWB=True, CAMERA_LOCK_AE=False), _picam2=fake)
    try:
        assert {'AwbEnable': False, 'ColourGains': (1.8, 1.5)} in fake.controls_set
    finally:
        cam.close()


def test_csi_ae_lock_freezes_measured_exposure(monkeypatch):
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_LOCK_AWB=False, CAMERA_LOCK_AE=True), _picam2=fake)
    try:
        assert {'AeEnable': False, 'ExposureTime': 19999,
                'AnalogueGain': 2.5} in fake.controls_set
        # exposed so a bench log can record what the run was shot at
        assert cam.locked == {'ExposureTime': 19999, 'AnalogueGain': 2.5}
    finally:
        cam.close()


def test_csi_ae_lock_skipped_when_metadata_lacks_keys(monkeypatch):
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2()
    fake.metadata = {'ColourGains': (1.8, 1.5)}      # sensor reports no exposure
    cam = CsiCamera(make_cfg(CAMERA_LOCK_AWB=True, CAMERA_LOCK_AE=True), _picam2=fake)
    try:
        assert all('AeEnable' not in c for c in fake.controls_set)
        assert cam.locked == {'ColourGains': (1.8, 1.5)}
    finally:
        cam.close()


def test_csi_relock_remeters_now_and_freezes_from_the_capture_thread(monkeypatch):
    # START on the course: auto goes back on immediately, and the capture
    # thread (not the caller) pins the new values once the warmup passes.
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2(frames=400, frame_delay=0.005)    # ~2 s of live frames
    cam = CsiCamera(make_cfg(CAMERA_LOCK_AWB=True, CAMERA_LOCK_AE=True), _picam2=fake)
    try:
        assert cam.locked['ExposureTime'] == 19999          # the boot lock
        fake.metadata = {'ColourGains': (2.0, 1.2),
                         'ExposureTime': 30000, 'AnalogueGain': 1.0}
        cam.relock()
        assert fake.controls_set[-1] == {'AwbEnable': True, 'AeEnable': True}
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and cam.locked.get('ExposureTime') != 30000:
            time.sleep(0.005)
        assert cam.locked == {'ColourGains': (2.0, 1.2),
                              'ExposureTime': 30000, 'AnalogueGain': 1.0}
        assert fake.controls_set[-1] == {'AwbEnable': False, 'ColourGains': (2.0, 1.2),
                                         'AeEnable': False, 'ExposureTime': 30000,
                                         'AnalogueGain': 1.0}
    finally:
        cam.close()


def test_csi_relock_is_a_no_op_with_both_locks_off():
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_LOCK_AWB=False, CAMERA_LOCK_AE=False), _picam2=fake)
    try:
        cam.relock()
        assert fake.controls_set == []
        assert cam.locked == {}
    finally:
        cam.close()


def test_csi_first_capture_failure_raises():
    with pytest.raises(RuntimeError):
        CsiCamera(make_cfg(), _picam2=FakePicamera2(first_frame_none=True))
