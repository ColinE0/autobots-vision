"""
Stop sign / traffic light detection.

make_detector(cfg) returns the backend named by cfg.DETECTOR_BACKEND:
  "classical"  HSV color masks + contour analysis (10-20 FPS on the Zero 2 W)
  "geometric"  Ryan's octagon-fit + glow-profile detectors, dropped in from
               autobots-vision (vision/detector_geometric.py); A/B at bring-up
  "yolo"       yolov8n via NCNN export (~1 FPS; A/B testing only)

Both produce a list of Detection(label, area_frac) per frame. Labels:
  "stop_sign", "red_light", "yellow_light", "green_light"

Classical pipeline per frame:
  1. downscale to DETECT_PROC_WIDTH (everything below runs on the small frame)
  2. skip the bottom DETECT_IGNORE_BOTTOM_FRAC entirely (floor tape, not signs)
  3. HSV inRange per color; red needs two hue bands (hue wraps at 180)
  4. contours; keep blobs >= DETECT_MIN_AREA_FRAC of the frame, roughly
     compact (extent/aspect gates reject streaks), largest blob per label
  5. red blobs split by WHITE CONTENT, not shape: a stop sign has white STOP
     text and border inside the red (>= STOPSIGN_WHITE_FRAC of its bounding
     box); a lit red lamp has none. A regular octagon has circularity 0.95,
     so shape alone cannot separate them. One red blob = one detection.

TemporalFilter does K-of-N confirmation (CONFIRM_FRAMES_K of the last
CONFIRM_FRAMES_N frames) so single-frame glints and motion blur do nothing.

The HSV ranges below were set for saturated course props under room light.
Verify with tools/test_camera.py in the actual demo room and widen if hits
are marginal. Known limit: a stop sign in deep shadow (letters no longer
read as white) classifies as red_light; consequence is a longer stop, never
a missed stop.
"""
from collections import deque, namedtuple

import cv2
import numpy as np

# The whole pipeline is small-image work where OpenCV's thread fan-out costs
# more than it saves, and the 50 Hz control / 100 Hz IR threads need cores.
cv2.setNumThreads(2)

# HSV ranges (OpenCV scale: H 0-180, S/V 0-255)
_LIGHT_S_MIN = 100
_LIGHT_V_MIN = 80
RED_BANDS = [((0, _LIGHT_S_MIN, _LIGHT_V_MIN), (10, 255, 255)),
             ((170, _LIGHT_S_MIN, _LIGHT_V_MIN), (180, 255, 255))]
YELLOW_BANDS = [((18, _LIGHT_S_MIN, _LIGHT_V_MIN), (35, 255, 255))]
GREEN_BANDS = [((40, _LIGHT_S_MIN, _LIGHT_V_MIN), (90, 255, 255))]
# White STOP text/border: nearly colorless and bright (S under 70, V over 170).
WHITE_LOW, WHITE_HIGH = (0, 0, 170), (180, 70, 255)


def _prep(bands):
    """Build the inRange ndarrays once at import, not once per frame."""
    return [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in bands]


_RED = _prep(RED_BANDS)
_YELLOW = _prep(YELLOW_BANDS)
_GREEN = _prep(GREEN_BANDS)
_WHITE_LOW = np.array(WHITE_LOW, np.uint8)
_WHITE_HIGH = np.array(WHITE_HIGH, np.uint8)


Detection = namedtuple('Detection', ['label', 'area_frac'])


def _mask(hsv, prepped):
    """OR together one or more prepared (low, high) HSV inRange masks."""
    out = None
    for low, high in prepped:
        m = cv2.inRange(hsv, low, high)
        out = m if out is None else cv2.bitwise_or(out, m)
    return out


class Detector:
    # Classical HSV detector. detect(frame_bgr) -> list of Detection
    def __init__(self, cfg):
        self.cfg = cfg
        self._kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame_bgr):
        cfg = self.cfg
        h, w = frame_bgr.shape[:2]
        if w != cfg.DETECT_PROC_WIDTH:
            scale = cfg.DETECT_PROC_WIDTH / float(w)
            frame_bgr = cv2.resize(frame_bgr,
                                   (cfg.DETECT_PROC_WIDTH, max(1, int(h * scale))))
        sh, sw = frame_bgr.shape[:2]
        frame_area = float(sh * sw)          # fractions stay relative to the FULL frame

        # Only the band above the floor cut can contain signs; converting and
        # masking just that slice does ~25% less work than blanking it out.
        cut = int(sh * (1.0 - cfg.DETECT_IGNORE_BOTTOM_FRAC))
        hsv = cv2.cvtColor(frame_bgr[:cut], cv2.COLOR_BGR2HSV)
        white = None                          # built only if a red blob needs it

        best = {}
        for color, prepped in (('red', _RED),
                               ('yellow_light', _YELLOW),
                               ('green_light', _GREEN)):
            mask = cv2.morphologyEx(_mask(hsv, prepped), cv2.MORPH_OPEN, self._kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                frac = area / frame_area
                if frac < cfg.DETECT_MIN_AREA_FRAC:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                extent = area / float(bw * bh)
                aspect = bw / float(bh)
                if extent < 0.45 or not (0.4 < aspect < 2.5):
                    continue                # streaks and edge glints, not props
                if color == 'red':
                    if white is None:
                        white = cv2.inRange(hsv, _WHITE_LOW, _WHITE_HIGH)
                    box_white = cv2.countNonZero(white[y:y + bh, x:x + bw])
                    white_frac = box_white / float(bw * bh)
                    label = ('stop_sign' if white_frac >= cfg.STOPSIGN_WHITE_FRAC
                             else 'red_light')
                else:
                    label = color
                if frac > best.get(label, 0.0):
                    best[label] = frac
        return [Detection(k, v) for k, v in best.items()]


class TemporalFilter:
    # K-of-N confirmation over per-frame detection lists. Call update() once
    # per processed frame; confirmed(label) goes True once the label showed up
    # in CONFIRM_FRAMES_K of the last CONFIRM_FRAMES_N frames. area(label) is
    # the most recent area_frac inside the window.
    def __init__(self, cfg):
        self._k = cfg.CONFIRM_FRAMES_K
        self._frames = deque(maxlen=cfg.CONFIRM_FRAMES_N)

    def update(self, detections):
        self._frames.append({d.label: d.area_frac for d in detections})

    def reset(self):
        self._frames.clear()

    def confirmed(self, label):
        return sum(1 for f in self._frames if label in f) >= self._k

    def area(self, label):
        for f in reversed(self._frames):
            if label in f:
                return f[label]
        return 0.0


class YoloDetector:
    # YOLOv8n (NCNN export) backend, ~1 FPS on the Zero 2 W, A/B testing only.
    # COCO "stop sign" maps directly; COCO "traffic light" carries no color, so
    # the box gets color-classified with the same HSV bands as the classical path.
    def __init__(self, cfg):
        from ultralytics import YOLO    # only installed for the A/B test
        self.cfg = cfg
        self._model = YOLO(cfg.YOLO_MODEL_DIR)

    def detect(self, frame_bgr):
        cfg = self.cfg
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        res = self._model.predict(frame_bgr, imgsz=cfg.YOLO_IMGSZ,
                                  conf=cfg.YOLO_CONF, verbose=False)[0]
        best = {}
        for box in res.boxes:
            name = res.names[int(box.cls)]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            frac = max(0, x2 - x1) * max(0, y2 - y1) / frame_area
            if name == 'stop sign':
                label = 'stop_sign'
            elif name == 'traffic light':
                label = _light_color(frame_bgr[y1:y2, x1:x2])
                if label is None:
                    continue
            else:
                continue
            if frac > best.get(label, 0.0):
                best[label] = frac
        return [Detection(k, v) for k, v in best.items()]


def _light_color(crop_bgr):
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    counts = {
        'red_light': cv2.countNonZero(_mask(hsv, _RED)),
        'yellow_light': cv2.countNonZero(_mask(hsv, _YELLOW)),
        'green_light': cv2.countNonZero(_mask(hsv, _GREEN)),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def make_detector(cfg):
    if cfg.DETECTOR_BACKEND == 'yolo':
        return YoloDetector(cfg)
    if cfg.DETECTOR_BACKEND == 'geometric':
        from vision.detector_geometric import GeometricDetector
        return GeometricDetector(cfg)
    return Detector(cfg)
