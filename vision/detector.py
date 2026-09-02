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
  3. HSV inRange per color; red needs two hue bands (hue wraps at 180).
     Yellow and green are masked at the lamp brightness floor LIGHT_V_MIN.
     Red is masked at the lower SIGN_V_MIN, because a printed stop sign is
     not a light source and must not have to clear a lamp's floor.
  4. contours; keep blobs >= DETECT_MIN_AREA_FRAC of the frame, roughly
     compact (extent/aspect gates reject streaks), largest blob per label
  5. GLOW test on every surviving blob: a lit lamp's colored pixels sit near
     clipping, so their mean V, 90th-percentile V and the share above a
     bright level all clear per-color floors (LAMP_GLOW). This is the test
     that separates a lit lamp from an unlit colored lens, a red poster, or
     a printed sign, and it runs BEFORE the white test below: a lit LED
     clips to white at its core, and that core used to read as STOP text.
     Bench 2026-09-02, exposure locked: six seconds of a CONFIRMED stop_sign
     at 14% on a lit red lamp, which on the robot ends the run at a red light.
  6. red that does not glow: stop_sign if >= STOPSIGN_WHITE_FRAC of its
     bounding box is white (STOP text and border), otherwise nothing. A
     regular octagon has circularity 0.95, so shape alone cannot separate a
     sign from a lamp. Yellow or green that does not glow is nothing: that is
     an unlit lens on the 3-lens module, which used to report all three
     colors at once.
  7. one lamp per frame. A traffic light shows one color at a time, so the
     largest glowing blob is reported and only if it beats the runner-up by
     LAMP_WINNER_RATIO; a near-tie reports no lamp that frame and the
     TemporalFilter absorbs the gap. A stop sign is reported alongside.

TemporalFilter does K-of-N confirmation (CONFIRM_FRAMES_K of the last
CONFIRM_FRAMES_N frames) so single-frame glints and motion blur do nothing.

The HSV ranges below were set for saturated course props under room light.
Verify with tools/test_camera.py in the actual demo room and widen if hits
are marginal. Known limit: a stop sign in deep shadow (letters no longer
read as white) is not reported at all. It used to classify as red_light,
which reads as "a longer stop" but is really a hold the robot cannot leave:
a stationary sign never clears, so the run sat there until the watchdog.
"""
from collections import deque, namedtuple

import cv2
import numpy as np

# The whole pipeline is small-image work where OpenCV's thread fan-out costs
# more than it saves, and the 50 Hz control / 100 Hz IR threads need cores.
cv2.setNumThreads(2)

# HSV ranges (OpenCV scale: H 0-180, S/V 0-255). The hues are fixed by
# physics; the saturation and brightness floors are the tuning knobs and
# live in config as LIGHT_S_MIN / LIGHT_V_MIN / SIGN_V_MIN.
def _light_bands(s_min, v_min):
    return {
        'red': [((0, s_min, v_min), (10, 255, 255)),
                ((170, s_min, v_min), (180, 255, 255))],
        'yellow': [((18, s_min, v_min), (35, 255, 255))],
        'green': [((40, s_min, v_min), (90, 255, 255))],
    }
# White STOP text/border: nearly colorless and bright (S under 70, V over 170).
WHITE_LOW, WHITE_HIGH = (0, 0, 170), (180, 70, 255)


def _prep(bands):
    """Build the inRange ndarrays once per detector, not once per frame."""
    return [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in bands]


def _prep_all(cfg):
    """Lamp-floor bands for every color (the YOLO crop classifier's set)."""
    raw = _light_bands(cfg.LIGHT_S_MIN, cfg.LIGHT_V_MIN)
    return {k: _prep(v) for k, v in raw.items()}
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


def _glows(v_vals, v_floor, gate):
    """True if a blob's colored pixels read like a lit lamp.

    v_vals: V channel of the blob's masked pixels. Only those at or above
    v_floor are judged, so the red blob (masked at the lower sign floor) is
    scored on the same population as yellow and green. gate is one
    LAMP_GLOW entry: (mean V, 90th-percentile V, bright share, bright level).
    """
    mean_min, peak_min, bright_frac, bright_v = gate
    v = v_vals[v_vals >= v_floor]
    if v.size == 0:
        return False
    return (float(v.mean()) >= mean_min
            and float(np.percentile(v, 90)) >= peak_min
            and float(np.mean(v >= bright_v)) >= bright_frac)


class Detector:
    # Classical HSV detector. detect(frame_bgr) -> list of Detection
    def __init__(self, cfg):
        self.cfg = cfg
        self._kernel = np.ones((3, 3), np.uint8)
        lamp = _light_bands(cfg.LIGHT_S_MIN, cfg.LIGHT_V_MIN)
        sign = _light_bands(cfg.LIGHT_S_MIN, cfg.SIGN_V_MIN)
        self._bands = {'red': _prep(sign['red']),
                       'yellow': _prep(lamp['yellow']),
                       'green': _prep(lamp['green'])}
        self._glow = {k: tuple(v) for k, v in cfg.LAMP_GLOW.items()}

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
        vch = hsv[:, :, 2]
        white = None                          # built only if a red blob needs it

        lamps = {}                            # color -> largest glowing blob
        sign = 0.0                            # largest stop sign
        for color in ('red', 'yellow', 'green'):
            mask = cv2.morphologyEx(_mask(hsv, self._bands[color]),
                                    cv2.MORPH_OPEN, self._kernel)
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
                box = (slice(y, y + bh), slice(x, x + bw))
                vals = vch[box][mask[box] > 0]
                if _glows(vals, cfg.LIGHT_V_MIN, self._glow[color]):
                    if frac > lamps.get(color, 0.0):
                        lamps[color] = frac
                    continue
                if color != 'red':
                    continue                # an unlit yellow/green lens
                if white is None:
                    white = cv2.inRange(hsv, _WHITE_LOW, _WHITE_HIGH)
                white_frac = cv2.countNonZero(white[box]) / float(bw * bh)
                if white_frac >= cfg.STOPSIGN_WHITE_FRAC and frac > sign:
                    sign = frac

        out = []
        if sign > 0.0:
            out.append(Detection('stop_sign', sign))
        if lamps:
            ranked = sorted(lamps.items(), key=lambda kv: kv[1], reverse=True)
            color, frac = ranked[0]
            if len(ranked) == 1 or frac >= ranked[1][1] * cfg.LAMP_WINNER_RATIO:
                out.append(Detection(color + '_light', frac))
        return out


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
        self._bands = _prep_all(cfg)

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
                label = _light_color(frame_bgr[y1:y2, x1:x2], self._bands)
                if label is None:
                    continue
            else:
                continue
            if frac > best.get(label, 0.0):
                best[label] = frac
        return [Detection(k, v) for k, v in best.items()]


def _light_color(crop_bgr, bands):
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    counts = {
        'red_light': cv2.countNonZero(_mask(hsv, bands['red'])),
        'yellow_light': cv2.countNonZero(_mask(hsv, bands['yellow'])),
        'green_light': cv2.countNonZero(_mask(hsv, bands['green'])),
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
