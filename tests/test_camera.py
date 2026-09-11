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

    def __init__(self, frames=1, first_frame_none=False, frame_delay=0.0, frame=None):
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
        self._served = frame

    def create_video_configuration(self, main=None, controls=None, raw=None,
                                   transform=None):
        return {'main': main, 'controls': controls, 'raw': raw,
                'transform': transform}

    def configure(self, cfg):
        self.video_config = cfg

    def start(self):
        self.started = True

    def capture_array(self, name='main', wait=None):
        if self._first_none:
            self._first_none = False
            return None
        if self._frames_left > 0:
            self._frames_left -= 1
            if self._frame_delay:
                time.sleep(self._frame_delay)
            return (self._served if self._served is not None
                    else np.zeros((240, 320, 3), np.uint8))
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
    cfg = make_cfg(CAMERA_EXPOSURE_US=None, CAMERA_WIDTH=320, CAMERA_HEIGHT=240, CAMERA_FPS=30,
                   CAMERA_AE_CONSTRAINT='highlight', CAMERA_EV=0.0)
    fake = FakePicamera2()
    cam = CsiCamera(cfg, _picam2=fake)
    try:
        assert fake.started
        # RGB888 is B,G,R in memory (libcamera names run backwards), which is
        # OpenCV's order; the detector's HSV math depends on this exact string.
        assert fake.video_config['main'] == {'size': (320, 240), 'format': 'RGB888'}
        # Metering controls ride in the stream configuration so the boot
        # lock and the START relock both meter under them (review 2026-09-05).
        assert fake.video_config['controls'] == {'FrameRate': 30.0,
                                                 'AeConstraintMode': 1,
                                                 'ExposureValue': 0.0}
    finally:
        cam.close()
    assert fake.closed


def test_csi_requests_the_full_frame_sensor_mode_by_default():
    # Without an explicit raw stream picamera2 picks the IMX219's 640x480
    # mode for a 320x240 main, a 1280x960 centre crop (flight Zero,
    # 2026-09-08). The lane model and the sign geometry both assume the
    # lens's real field of view, so the mode is named, not inferred.
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(), _picam2=fake)
    try:
        assert fake.video_config['raw'] == {'size': (1640, 1232)}
        assert cam.sensor_size == (1640, 1232)
    finally:
        cam.close()


def test_csi_sensor_size_none_leaves_the_mode_to_picamera2():
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_SENSOR_SIZE=None), _picam2=fake)
    try:
        assert fake.video_config['raw'] is None
        assert cam.sensor_size is None
    finally:
        cam.close()


@pytest.mark.parametrize('name,value', [('normal', 0), ('highlight', 1), ('shadows', 2)])
def test_csi_ae_constraint_names_map_to_libcamera_enum(name, value):
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_AE_CONSTRAINT=name), _picam2=fake)
    try:
        assert fake.video_config['controls']['AeConstraintMode'] == value
    finally:
        cam.close()


def test_csi_rejects_unknown_ae_constraint():
    with pytest.raises(ValueError):
        CsiCamera(make_cfg(CAMERA_AE_CONSTRAINT='bright'), _picam2=FakePicamera2())


def test_csi_exposure_value_is_floored_at_minus_one_stop():
    # Below -1 stop a V 180 printed sign drops under SIGN_V_MIN; the floor
    # keeps a typo in config from making every sign invisible.
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_EV=-2.5), _picam2=fake)
    try:
        assert fake.video_config['controls']['ExposureValue'] == -1.0
    finally:
        cam.close()
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_EV=-0.5), _picam2=fake)
    try:
        assert fake.video_config['controls']['ExposureValue'] == -0.5
    finally:
        cam.close()


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
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=None, CAMERA_LOCK_AWB=False, CAMERA_LOCK_AE=True), _picam2=fake)
    try:
        assert {'AeEnable': False, 'ExposureTime': 19999,
                'AnalogueGain': 2.5} in fake.controls_set
        # exposed so a bench log can record what the run was shot at
        assert cam.locked == {'ExposureTime': 19999, 'AnalogueGain': 2.5}
    finally:
        cam.close()


def test_csi_ae_lock_at_the_frame_ceiling_warns(monkeypatch, capsys):
    # AE out of shutter (a dark room): the detector floors were tuned under
    # room light, so the lock says so and exposes a flag for the pilot log.
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2()
    fake.metadata['ExposureTime'] = 33000        # of a 33333 us frame at 30 fps
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=None, CAMERA_LOCK_AWB=False, CAMERA_LOCK_AE=True,
                             CAMERA_FPS=30), _picam2=fake)
    try:
        assert cam.exposure_at_ceiling
        assert 'frame ceiling' in capsys.readouterr().out
    finally:
        cam.close()


def test_csi_ae_lock_under_the_ceiling_is_quiet(monkeypatch, capsys):
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    cam = CsiCamera(make_cfg(CAMERA_LOCK_AWB=False, CAMERA_LOCK_AE=True,
                             CAMERA_FPS=30), _picam2=FakePicamera2())
    try:
        assert not cam.exposure_at_ceiling
        assert 'frame ceiling' not in capsys.readouterr().out
    finally:
        cam.close()


def test_csi_ae_lock_skipped_when_metadata_lacks_keys(monkeypatch):
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2()
    fake.metadata = {'ColourGains': (1.8, 1.5)}      # sensor reports no exposure
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=None, CAMERA_LOCK_AWB=True, CAMERA_LOCK_AE=True), _picam2=fake)
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
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=None, CAMERA_LOCK_AWB=True, CAMERA_LOCK_AE=True), _picam2=fake)
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
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=None, CAMERA_LOCK_AWB=False, CAMERA_LOCK_AE=False), _picam2=fake)
    try:
        cam.relock()
        assert fake.controls_set == []
        assert cam.locked == {}
    finally:
        cam.close()


def test_csi_first_frame_timeout_raises_instead_of_hanging():
    # Flight Zero 2026-09-08: sensor enumerated over i2c but the CSI data
    # lanes delivered nothing, and CsiCamera hung forever on its first
    # capture_array. picamera2 raises TimeoutError when wait= is a number.
    class NoFrames(FakePicamera2):
        def capture_array(self, name='main', wait=None):
            assert wait is not None and wait > 0, 'first capture must carry a timeout'
            raise TimeoutError
    fake = NoFrames()
    with pytest.raises(RuntimeError, match='ribbon'):
        CsiCamera(make_cfg(), _picam2=fake)


def test_csi_first_capture_failure_raises():
    with pytest.raises(RuntimeError):
        CsiCamera(make_cfg(), _picam2=FakePicamera2(first_frame_none=True))



def test_csi_rotates_every_frame_when_the_mount_is_upside_down():
    # The ISP transform was tried first and is silently dropped when the raw
    # stream is pinned, so the frames the CALLER gets are what must be checked,
    # not what was asked of libcamera.
    marked = np.zeros((240, 320, 3), np.uint8)
    marked[0, 0] = 255                      # top-left corner only
    fake = FakePicamera2(frames=1, frame=marked)
    cam = CsiCamera(make_cfg(CAMERA_ROTATE_180=True), _picam2=fake)
    _, frame = cam.read()
    cam.close()
    assert frame[239, 319].tolist() == [255, 255, 255], "mark must move to bottom-right"
    assert frame[0, 0].tolist() == [0, 0, 0]
    assert cam.rotated is True


def test_csi_leaves_frames_alone_when_the_mount_is_upright():
    marked = np.zeros((240, 320, 3), np.uint8)
    marked[0, 0] = 255
    fake = FakePicamera2(frames=1, frame=marked)
    cam = CsiCamera(make_cfg(CAMERA_ROTATE_180=False), _picam2=fake)
    _, frame = cam.read()
    cam.close()
    assert frame[0, 0].tolist() == [255, 255, 255]
    assert cam.rotated is False

def test_fixed_exposure_pins_from_the_configuration_not_a_later_lock(monkeypatch):
    # The FIRST frame has to be at the fixed value. Pinning it after start would
    # leave a metered frame or two at the head of every run.
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=4000, CAMERA_ANALOGUE_GAIN=2.0),
                    _picam2=fake)
    controls = fake.video_config['controls']
    cam.close()
    assert controls['AeEnable'] is False
    assert controls['ExposureTime'] == 4000
    assert controls['AnalogueGain'] == 2.0
    assert cam.locked['ExposureTime'] == 4000


def test_fixed_exposure_never_meters_the_room(monkeypatch):
    # The metered value in the fake's metadata (19999 us) must not reach the
    # camera: a fixed exposure that re-meters is just AE with extra steps.
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=4000, CAMERA_LOCK_AE=True,
                             CAMERA_LOCK_AWB=False), _picam2=fake)
    cam.close()
    assert all('ExposureTime' not in c for c in fake.controls_set)
    assert cam.exposure_at_ceiling is False


def test_relock_leaves_a_pinned_exposure_alone(monkeypatch):
    # START re-meters the scene when AE owns exposure. With a pinned exposure
    # that would hand the run back to the room mid-course.
    monkeypatch.setattr(camera_mod, '_LOCK_WARMUP_S', 0.0)
    fake = FakePicamera2()
    cam = CsiCamera(make_cfg(CAMERA_EXPOSURE_US=4000, CAMERA_LOCK_AE=True,
                             CAMERA_LOCK_AWB=True), _picam2=fake)
    fake.controls_set.clear()
    cam.relock()
    cam.close()
    assert all(c.get('AeEnable') is not True for c in fake.controls_set)


def test_fixed_exposure_rejects_nonsense():
    with pytest.raises(ValueError, match='microseconds'):
        camera_mod._fixed_exposure_controls(make_cfg(CAMERA_EXPOSURE_US=0))
    with pytest.raises(ValueError, match='below 1.0'):
        camera_mod._fixed_exposure_controls(
            make_cfg(CAMERA_EXPOSURE_US=4000, CAMERA_ANALOGUE_GAIN=0.5))


def test_no_fixed_exposure_keeps_the_metering_path():
    assert camera_mod._fixed_exposure_controls(make_cfg(CAMERA_EXPOSURE_US=None)) == {}


def test_shipped_config_still_carries_the_rotation_flag():
    # Regression guard. CAMERA_ROTATE_180 was deleted once by an unrelated edit
    # (b4e56c9) and nothing caught it, because camera.py reads it with a getattr
    # default of False: the camera just went back to being upside down. Assert on
    # the real config module, not on make_cfg's copy.
    import config as shipped
    assert getattr(shipped, 'CAMERA_ROTATE_180', None) is True


def test_shipped_config_pins_the_bench_exposure():
    # 4000 us was measured, not guessed (2026-09-11 sweep). Guarded for the
    # same reason as the rotation flag: a config value that silently reverts
    # to None hands the run back to the room and nothing else would notice.
    import config as shipped
    assert shipped.CAMERA_EXPOSURE_US == 4000
    assert shipped.CAMERA_ANALOGUE_GAIN == 1.0
