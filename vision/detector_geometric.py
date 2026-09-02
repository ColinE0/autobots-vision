"""
Stop sign / traffic light detection, geometric backend.

GeometricDetector wraps the two detectors in this package
(vision/stop_sign_detector.py, vision/traffic_light_detector.py) behind the
same contract the robot's vision thread already speaks:

    detect(frame_bgr) -> list of Detection(label, area_frac)

Labels: "stop_sign", "red_light", "yellow_light", "green_light". This file
plus the two detector modules drop into the robot repo's vision/ unchanged;
the robot selects the backend by name in make_detector.

Per frame: downscale to cfg.DETECT_PROC_WIDTH, then ONE blur + HSV
conversion shared by both detectors (the Zero 2 W cannot afford two), then
the arbitration rule from the bench harness: if the stop sign and the
traffic light land on the same object, the traffic light wins, because a lit
red lamp is the object most likely to fool the octagon test.

area_frac is the detection's bounding box over the frame, which reads a
little larger than the classical backend's contour-area fraction; tune the
ACT_AREA_FRAC thresholds per backend at bring-up, don't share them.

Temporal smoothing is deliberately absent here. The robot's TemporalFilter
(K-of-N confirmation) sits downstream, exactly as it does for the classical
backend.
"""
from collections import namedtuple

import cv2

from vision.stop_sign_detector import detect_stop_sign
from vision.traffic_light_detector import detect_traffic_light

# The whole pipeline is small-image work where OpenCV's thread fan-out costs
# more than it saves, and the 50 Hz control / 100 Hz IR threads need cores.
cv2.setNumThreads(2)

Detection = namedtuple('Detection', ['label', 'area_frac'])

_LIGHT_LABELS = {
    'RED': 'red_light',
    'YELLOW': 'yellow_light',
    'GREEN': 'green_light',
}


def boxes_represent_same_object(box1, box2):
    """Return True if two detection boxes likely refer to the same object."""

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    if right <= left or bottom <= top:
        return False

    overlap_area = (right - left) * (bottom - top)

    area1 = w1 * h1
    area2 = w2 * h2

    if area1 <= 0 or area2 <= 0:
        return False

    # Compare overlap against the smaller detection box.
    smaller_area = min(area1, area2)
    overlap_ratio = overlap_area / smaller_area

    # Calculate box centers.
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2

    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2

    # Check whether either detection center is inside the other box.
    center1_inside_box2 = (
        x2 <= center1_x <= x2 + w2
        and y2 <= center1_y <= y2 + h2
    )

    center2_inside_box1 = (
        x1 <= center2_x <= x1 + w1
        and y1 <= center2_y <= y1 + h1
    )

    return (
        overlap_ratio >= 0.50
        or center1_inside_box2
        or center2_inside_box1
    )


class GeometricDetector:
    # detect(frame_bgr) -> list of Detection, same contract as the robot's
    # classical Detector.
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, frame_bgr):
        cfg = self.cfg
        h, w = frame_bgr.shape[:2]

        if w != cfg.DETECT_PROC_WIDTH:
            scale = cfg.DETECT_PROC_WIDTH / float(w)
            frame_bgr = cv2.resize(
                frame_bgr,
                (cfg.DETECT_PROC_WIDTH, max(1, int(h * scale)))
            )

        sh, sw = frame_bgr.shape[:2]
        frame_area = float(sh * sw)

        # One blur + HSV conversion, shared by both detectors.
        blurred = cv2.GaussianBlur(frame_bgr, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        sign = detect_stop_sign(frame_bgr, hsv)
        light = detect_traffic_light(frame_bgr, hsv)

        # Traffic-light detection takes priority if both detectors appear to
        # be looking at the same physical object.
        if sign is not None and light is not None:
            if boxes_represent_same_object(sign["box"], light["box"]):
                sign = None

        detections = []

        if sign is not None:
            _, _, bw, bh = sign["box"]
            detections.append(
                Detection('stop_sign', bw * bh / frame_area))

        if light is not None:
            _, _, bw, bh = light["box"]
            detections.append(
                Detection(_LIGHT_LABELS[light["color"]], bw * bh / frame_area))

        return detections
