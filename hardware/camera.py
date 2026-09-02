"""
Camera capture. Two backends behind make_camera(), picked by CAMERA_BACKEND:

  "csi"  CsiCamera: Arducam 8MP IMX219 on the CSI ribbon via picamera2.
         The ISP hands over frames already at CAMERA_WIDTH x CAMERA_HEIGHT
         in BGR order, so there is no MJPG decode and no USB stack in the
         path. The robot camera as of 2026-07-20.
  "usb"  UsbCamera: DFRobot FIT0701 or any V4L2 webcam (the bench spare).

Both free-run a capture thread that keeps only the newest frame, so the
vision loop always processes the freshest image and never blocks the control
loop. read() returns (frame_id, frame). Callers compare frame_id to skip
work when no new frame has arrived.
"""
import threading
import time

import cv2

# CSI AWB/AE lock: seconds of settling before the values are frozen (only
# when CAMERA_LOCK_AWB / CAMERA_LOCK_AE is set), at startup and on relock().
_LOCK_WARMUP_S = 1.0


class UsbCamera:
    # V4L2/OpenCV capture (the pre-2026-07-20 robot camera, kept as the spare).
    def __init__(self, cfg):
        self.cfg = cfg
        self._cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_V4L2)
        if cfg.CAMERA_USE_MJPG:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.CAMERA_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Camera index {cfg.CAMERA_INDEX} did not open. "
                f"Check the USB cable and `ls /dev/video*`.")
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError('Camera opened but the first read failed.')
        self._lock = threading.Lock()
        self._frame = frame
        self._frame_id = 1
        self._run = True
        self._thread = threading.Thread(target=self._loop, name='camera', daemon=True)
        self._thread.start()

    def _loop(self):
        while self._run:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
            else:
                time.sleep(0.05)     # transient USB hiccup; keep trying

    def read(self):
        """Returns (frame_id, newest BGR frame). Never blocks."""
        with self._lock:
            return self._frame_id, self._frame

    def close(self):
        self._run = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class CsiCamera:
    # picamera2 capture for the Arducam IMX219 (or any libcamera CSI module).
    def __init__(self, cfg, _picam2=None):
        self.cfg = cfg
        if _picam2 is not None:          # tests inject a fake here
            self._picam = _picam2
        else:
            from picamera2 import Picamera2   # apt: python3-picamera2
            self._picam = Picamera2()
        # libcamera format names read backwards from memory order: "RGB888"
        # is B,G,R in memory, which is exactly OpenCV's BGR. Do not "correct"
        # it to BGR888; that one comes out RGB and every hue in the detector
        # shifts. Bring-up check: a real stop sign must label stop_sign or
        # red_light in tools/test_camera.py, never blue/nothing.
        stream = self._picam.create_video_configuration(
            main={'size': (cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT),
                  'format': 'RGB888'},
            controls={'FrameRate': float(cfg.CAMERA_FPS)})
        self._picam.configure(stream)
        self._picam.start()
        frame = self._picam.capture_array('main')
        if frame is None or frame.size == 0:
            raise RuntimeError(
                'CSI camera started but the first capture failed. Check the '
                'flex ribbon seating and `rpicam-hello --list-cameras`.')
        self.locked = {}
        self._relock_at = None
        if cfg.CAMERA_LOCK_AWB or cfg.CAMERA_LOCK_AE:
            # Freeze white balance and/or exposure once they settle. AWB left
            # live lets a big red/green prop drag every hue with it mid-run.
            # AE left live is worse for detection: re-aiming the camera
            # re-meters the scene, so whether a lamp clears a brightness gate
            # depends on framing rather than on the lamp. Bench 2026-09-01,
            # a red lamp only detected after the camera was nudged upward.
            time.sleep(_LOCK_WARMUP_S)
            self._freeze()
        self._lock = threading.Lock()
        self._frame = frame
        self._frame_id = 1
        self._run = True
        self._thread = threading.Thread(target=self._loop, name='camera', daemon=True)
        self._thread.start()

    def _freeze(self):
        """Read the sensor's current AWB gains / exposure and pin them."""
        cfg = self.cfg
        md = self._picam.capture_metadata()
        controls = {}
        # Only freeze what the sensor actually reports. A missing key is
        # not worth crashing the camera over; the run just stays auto for
        # that control, and the header records what was locked.
        if cfg.CAMERA_LOCK_AWB and 'ColourGains' in md:
            controls['AwbEnable'] = False
            controls['ColourGains'] = md['ColourGains']
        if cfg.CAMERA_LOCK_AE and {'ExposureTime', 'AnalogueGain'} <= md.keys():
            controls['AeEnable'] = False
            controls['ExposureTime'] = md['ExposureTime']
            controls['AnalogueGain'] = md['AnalogueGain']
        if controls:
            self._picam.set_controls(controls)
        # Keep the frozen values: two runs are only comparable if they
        # were taken at the same exposure, so callers can log them.
        self.locked = {k: v for k, v in controls.items()
                       if k not in ('AwbEnable', 'AeEnable')}

    def relock(self):
        """Re-meter the scene and freeze again, without blocking the caller.

        The startup lock pins whatever the camera saw one second after
        power-up: the floor, a hand, the bench. The pilot calls this on
        START, when the robot is on the course and pointing down it, so the
        run's exposure comes from the course. Auto is re-enabled right now;
        the capture thread pins the new values _LOCK_WARMUP_S later (the PRD
        countdown is 5 s, so the lock lands before the wheels turn).
        """
        cfg = self.cfg
        auto = {}
        if cfg.CAMERA_LOCK_AWB:
            auto['AwbEnable'] = True
        if cfg.CAMERA_LOCK_AE:
            auto['AeEnable'] = True
        if not auto:
            return
        self._picam.set_controls(auto)
        self._relock_at = time.monotonic() + _LOCK_WARMUP_S

    def _loop(self):
        while self._run:
            try:
                frame = self._picam.capture_array('main')
            except Exception:
                frame = None             # stopped mid-capture, or a glitch
            if frame is None:
                if self._run:
                    time.sleep(0.05)     # transient capture error; keep trying
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            if self._relock_at is not None and time.monotonic() >= self._relock_at:
                self._relock_at = None
                try:
                    self._freeze()
                except Exception:
                    pass                 # stays auto; the run goes on

    def read(self):
        """Returns (frame_id, newest BGR frame). Never blocks."""
        with self._lock:
            return self._frame_id, self._frame

    def close(self):
        self._run = False
        try:
            self._picam.stop()           # unblocks a capture_array in flight
        except Exception:
            pass
        self._thread.join(timeout=1.0)
        try:
            self._picam.close()
        except Exception:
            pass


def make_camera(cfg):
    if cfg.CAMERA_BACKEND == 'csi':
        return CsiCamera(cfg)
    if cfg.CAMERA_BACKEND == 'usb':
        return UsbCamera(cfg)
    raise ValueError(
        f"CAMERA_BACKEND={cfg.CAMERA_BACKEND!r} is not a backend; use 'csi' or 'usb'")
