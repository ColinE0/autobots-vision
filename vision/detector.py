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
  3. HSV inRange per color; red needs two hue bands (hue wraps at 180). The
     red and yellow bands meet (red to hue 11, yellow from 12) so an orange
     yellow lamp under a warm white-balance lock cannot fall between them
     and vanish. Yellow and green are masked at the lamp brightness floor
     LIGHT_V_MIN. Red is masked at the lower SIGN_V_MIN, because a printed
     stop sign is not a light source and must not have to clear a lamp's floor.
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
     Optional spread veto (LAMP_MIN_V_SPREAD, off by default): a lamp falls
     off from centre to rim, a printed sign is flat, so a blob whose V spread
     (p90 minus p10) is under the knob is not a lamp however bright it is.
     Rendered frames 2026-09-07: lamps 61 to 75, flat signs 0 to 4. Turn it
     on (30) once a real lamp's spread has been read off a saved frame; a
     misfire on a real lamp would miss a red light.
  6. CLIPPED-CORE test on a blob that failed the glow test. A blown-out lamp
     (close, or metered under a long locked exposure) clips to white almost
     to its rim. The saturation floor in step 3 deletes that white core, so
     the glow test only ever sees the thin coloured ring and the halo, whose
     mean V lands under the red floor. The glow evidence is the white the
     colour mask threw away, so judge the white's SHAPE inside the blob:
     one round (LAMP_CORE_ASPECT), filled (LAMP_CORE_MIN_EXTENT), central
     (LAMP_CORE_CENTER_TOL) component covering >= LAMP_CORE_MIN_FRAC of the
     blob's box is an LED core, and the blob is a lamp of its colour. STOP
     text is a wide band and a sign's white border is hollow, so a printed
     sign never passes. Review 2026-09-05, synthetic frames: without this
     step every blown red lamp read as stop_sign, at 7% to 32% of the frame.
  7. red that neither glows nor has a core: stop_sign if >= STOPSIGN_WHITE_FRAC
     of the blob's interior is white (STOP text and border), otherwise nothing.
     A regular octagon has circularity 0.95, so shape alone cannot separate a
     sign from a lamp. Yellow or green that does not glow is nothing: that is
     an unlit lens on the 3-lens module, which used to report all three
     colors at once.
     WHITE is judged inside the FILLED red contour, not the bounding box (a
     round lamp's box has background in its corners, and a white wall there
     read as STOP text), and RELATIVE to the blob's own red: S under WHITE_S_MAX
     and V >= STOPSIGN_WHITE_REL x the median V of the red pixels, floored at
     STOPSIGN_WHITE_V_MIN and capped at STOPSIGN_WHITE_V_CAP so clipped
     letters still count. A printed sign is a reflector: a darker exposure
     dims its letters and its red together, so a fixed white floor lost the
     sign at 0.7x the tuning exposure (review 2026-09-07), which is exactly
     where the camera's highlight AE constraint puts it with a lamp in frame.
  8. one lamp per frame. A traffic light shows one color at a time, so the
     largest glowing blob is reported and only if it beats the runner-up by
     LAMP_WINNER_RATIO; a near-tie reports no lamp that frame and the
     TemporalFilter absorbs the gap. A stop sign is reported alongside.

TemporalFilter does K-of-N confirmation (CONFIRM_FRAMES_K of the last
CONFIRM_FRAMES_N frames) so single-frame glints and motion blur do nothing.

The HSV ranges below were set for saturated course props under room light.
Verify with tools/test_camera.py in the actual demo room and widen if hits
are marginal. Known limit: a stop sign so dark that its red drops under
SIGN_V_MIN is not reported at all. It used to classify as red_light, which
reads as "a longer stop" but is really a hold the robot cannot leave: a
stationary sign never clears, so the run sat there until the watchdog.
"""
from collections import deque, namedtuple

import cv2
import numpy as np

# The whole pipeline is small-image work where OpenCV's thread fan-out costs
# more than it saves, and the 50 Hz control / 100 Hz IR threads need cores.
cv2.setNumThreads(2)

# HSV ranges (OpenCV scale: H 0-180, S/V 0-255). The hues are fixed by
# physics; the saturation and brightness floors are the tuning knobs and
# live in config as LIGHT_S_MIN / LIGHT_V_MIN / SIGN_V_MIN. Red and yellow
# are adjacent on purpose (11 | 12): a gap between them is a colour that
# reads as nothing, and a yellow lamp read as nothing is driven through.
def _light_bands(s_min, v_min):
    return {
        'red': [((0, s_min, v_min), (11, 255, 255)),
                ((170, s_min, v_min), (180, 255, 255))],
        'yellow': [((12, s_min, v_min), (35, 255, 255))],
        'green': [((40, s_min, v_min), (90, 255, 255))],
    }
# White STOP text/border is nearly colorless: S under this. Its brightness
# floor is relative to the blob (step 7 above), not a constant.
WHITE_S_MAX = 70


def _prep(bands):
    """Build the inRange ndarrays once per detector, not once per frame."""
    return [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in bands]


def _prep_all(cfg):
    """Lamp-floor bands for every color (the YOLO crop classifier's set)."""
    raw = _light_bands(cfg.LIGHT_S_MIN, cfg.LIGHT_V_MIN)
    return {k: _prep(v) for k, v in raw.items()}


Detection = namedtuple('Detection', ['label', 'area_frac'])


def _mask(hsv, prepped):
    """OR together one or more prepared (low, high) HSV inRange masks."""
    out = None
    for low, high in prepped:
        m = cv2.inRange(hsv, low, high)
        out = m if out is None else cv2.bitwise_or(out, m)
    return out


def _glows(v_vals, v_floor, gate, min_spread=0):
    """True if a blob's colored pixels read like a lit lamp.

    v_vals: V channel of the blob's masked pixels. Only those at or above
    v_floor are judged, so the red blob (masked at the lower sign floor) is
    scored on the same population as yellow and green. gate is one
    LAMP_GLOW entry: (mean V, 90th-percentile V, bright share, bright level).
    min_spread > 0 adds the flatness veto: p90 minus p10 of the same pixels
    must reach it, because a lamp falls off toward its rim and a printed
    surface does not.
    """
    mean_min, peak_min, bright_frac, bright_v = gate
    v = v_vals[v_vals >= v_floor]
    if v.size == 0:
        return False
    p10, p90 = np.percentile(v, (10, 90))
    if min_spread > 0 and (p90 - p10) < min_spread:
        return False
    return (float(v.mean()) >= mean_min
            and float(p90) >= peak_min
            and float(np.mean(v >= bright_v)) >= bright_frac)


def _white_inside(hsv_box, inside, red_v, cfg):
    """White pixels inside a red blob, as a 0/255 mask the size of its box.

    inside: the blob's filled contour, so background never counts. The
    brightness floor follows the blob's own red (median V x STOPSIGN_WHITE_REL,
    at least STOPSIGN_WHITE_V_MIN, at most STOPSIGN_WHITE_V_CAP): sign letters
    and sign red are lit by the same light and dim together.
    """
    floor = cfg.STOPSIGN_WHITE_REL * float(np.median(red_v))
    floor = min(cfg.STOPSIGN_WHITE_V_CAP, max(cfg.STOPSIGN_WHITE_V_MIN, floor))
    # V is 8-bit: compare against an int inside its range. An out-of-range
    # float here crashed numpy 2 outright on the laptop (access violation).
    floor = int(min(255, max(0, round(floor))))
    white = ((hsv_box[:, :, 1] < WHITE_S_MAX) & (hsv_box[:, :, 2] >= floor)
             & (inside > 0))
    return white.astype(np.uint8) * 255


def _clipped_core(white_box, min_frac, aspect, min_extent, center_tol):
    """True if the white inside a blob's box is one round, filled, central
    mass: the clipped core of a lit LED.

    white_box: the white mask cropped to the blob's bounding box. Only the
    largest connected component is judged. STOP text is a 3.5:1 band, a
    sign's white border is a hollow ring, and neither sits as a single blob
    at the centre, so a printed sign fails on aspect, extent or position.
    """
    bh, bw = white_box.shape
    n, _, stats, cents = cv2.connectedComponentsWithStats(white_box, connectivity=8)
    if n < 2:
        return False                        # no white at all
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    _, _, w, h, area = stats[i]
    if area < min_frac * bw * bh:
        return False
    if not (aspect[0] < w / float(h) < aspect[1]):
        return False
    if area / float(w * h) < min_extent:
        return False
    cx, cy = cents[i]
    return (abs(cx - bw / 2.0) <= center_tol * bw
            and abs(cy - bh / 2.0) <= center_tol * bh)


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
        self._core = (cfg.LAMP_CORE_MIN_FRAC, tuple(cfg.LAMP_CORE_ASPECT),
                      cfg.LAMP_CORE_MIN_EXTENT, cfg.LAMP_CORE_CENTER_TOL)

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
        spread = getattr(cfg, 'LAMP_MIN_V_SPREAD', 0)

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
                if _glows(vals, cfg.LIGHT_V_MIN, self._glow[color], spread):
                    if frac > lamps.get(color, 0.0):
                        lamps[color] = frac
                    continue
                # Everything below judges the white INSIDE this blob: its
                # filled contour, so a white wall behind a lamp stays out.
                inside = np.zeros((bh, bw), np.uint8)
                cv2.drawContours(inside, [c - (x, y)], -1, 255, cv2.FILLED)
                white = _white_inside(hsv[box], inside, vals, cfg)
                # A blown-out lamp: its glow evidence is the white core the
                # colour mask threw away, so judge the white's SHAPE.
                if _clipped_core(white, *self._core):
                    if frac > lamps.get(color, 0.0):
                        lamps[color] = frac
                    continue
                if color != 'red':
                    continue                # an unlit yellow/green lens
                white_frac = (cv2.countNonZero(white)
                              / float(max(1, cv2.countNonZero(inside))))
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
