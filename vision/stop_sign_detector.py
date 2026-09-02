import cv2
import numpy as np
import math


# HSV ranges used to isolate red pixels
RED1_LOWER = np.array([0, 120, 70])
RED1_UPPER = np.array([10, 255, 255])

RED2_LOWER = np.array([170, 120, 70])
RED2_UPPER = np.array([180, 255, 255])


# Basic stop-sign size and shape requirements
STOP_MIN_AREA = 0.00015
STOP_MAX_AREA = 0.15

STOP_MIN_ASPECT = 0.70
STOP_MAX_ASPECT = 1.40

MIN_EXTENT = 0.40


# Ideal interior angle of a regular octagon
IDEAL_OCTAGON_ANGLE = 135.0

# Allowed variation in octagon geometry
MAX_ANGLE_ERROR = 24.3
MAX_SIDE_LENGTH_RATIO = 1.75


# Requirements for normal or close stop signs
NORMAL_MIN_SIDE_PIXELS = 6.0
NORMAL_MIN_OCTAGON_SCORE = 0.65
NORMAL_MIN_SHAPE_MATCH = 0.94


# More forgiving requirements for small or distant stop signs
FAR_OBJECT_AREA_RATIO = 0.002

FAR_MIN_SIDE_PIXELS = 2.5
FAR_MIN_OCTAGON_SCORE = 0.52
FAR_MIN_SHAPE_MATCH = 0.86


# Reject red objects that are almost perfectly circular
MAX_CIRCLE_FILL = 0.95


# Brightness requirements used to reject glowing red traffic lights
RED_LIGHT_MEAN_VALUE = 205
RED_LIGHT_PEAK_VALUE = 235
RED_LIGHT_BRIGHT_RATIO = 0.30
RED_LIGHT_BRIGHT_PIXEL = 220


# Kernels used to clean the red mask
OPEN_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_KERNEL = np.ones((5, 5), np.uint8)


def get_roi_mask(frame, top_cutoff=0.05, bottom_cutoff=0.80):
    """Restrict detection to the useful part of the frame."""

    h, w = frame.shape[:2]

    # Start with a black mask the same size as the frame
    mask = np.zeros((h, w), dtype=np.uint8)

    ymin = int(h * top_cutoff)
    ymax = int(h * bottom_cutoff)

    # White pixels represent the area where detection is allowed
    mask[ymin:ymax, :] = 255

    return mask


def clean_mask(mask):
    """Remove noise and reconnect nearby red regions."""

    # Opening removes small isolated pixels
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        OPEN_KERNEL
    )

    # Closing fills small gaps inside detected regions
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        CLOSE_KERNEL
    )

    return mask


def get_red_mask(hsv):
    """Create one mask containing all detected red pixels."""

    # Red appears at both ends of the HSV hue range
    red1 = cv2.inRange(
        hsv,
        RED1_LOWER,
        RED1_UPPER
    )

    red2 = cv2.inRange(
        hsv,
        RED2_LOWER,
        RED2_UPPER
    )

    # Combine both red ranges into one mask
    red_mask = cv2.bitwise_or(
        red1,
        red2
    )

    return clean_mask(red_mask)


def looks_like_red_light(hsv, red_mask):
    """Return True if a red region behaves like a glowing lamp."""

    # Extract brightness values only from detected red pixels
    values = hsv[:, :, 2][red_mask > 0]

    if values.size == 0:
        return False

    # Average brightness of the red region
    mean_value = float(
        np.mean(values)
    )

    # Brightness reached by the brightest 10% of red pixels
    peak_value = float(
        np.percentile(values, 90)
    )

    # Percentage of red pixels that are extremely bright
    bright_ratio = float(
        np.mean(
            values >= RED_LIGHT_BRIGHT_PIXEL
        )
    )

    # A region passing all three tests is treated as a red light
    return (
        mean_value >= RED_LIGHT_MEAN_VALUE
        and peak_value >= RED_LIGHT_PEAK_VALUE
        and bright_ratio >= RED_LIGHT_BRIGHT_RATIO
    )


def get_angle(p1, p2, p3):
    """Calculate the interior angle at point p2."""

    # Create vectors from the middle point to its two neighbors
    v1 = np.array(
        [
            p1[0] - p2[0],
            p1[1] - p2[1]
        ],
        dtype=np.float32
    )

    v2 = np.array(
        [
            p3[0] - p2[0],
            p3[1] - p2[1]
        ],
        dtype=np.float32
    )

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 <= 0 or norm2 <= 0:
        return 0.0

    # Dot-product formula gives the angle between the two vectors
    cosine = np.dot(v1, v2) / (
        norm1 * norm2
    )

    # Prevent small numerical errors from breaking acos()
    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )


def score_octagon(approx, min_side_pixels):
    """Score how closely an 8-sided polygon resembles an octagon."""

    # approx contains the corner points of the simplified polygon
    if len(approx) != 8:
        return 0.0, {}

    # A stop sign should form an outward-facing convex polygon
    if not cv2.isContourConvex(approx):
        return 0.0, {}

    # Convert OpenCV's point format into simple (x, y) coordinates
    points = [
        tuple(point[0])
        for point in approx
    ]

    angles = []

    # Calculate the interior angle at each of the 8 corners
    for i in range(8):
        previous_point = points[(i - 1) % 8]
        current_point = points[i]
        next_point = points[(i + 1) % 8]

        angle = get_angle(
            previous_point,
            current_point,
            next_point
        )

        angles.append(angle)

    # Compare every measured angle to the ideal 135-degree angle
    angle_errors = [
        abs(angle - IDEAL_OCTAGON_ANGLE)
        for angle in angles
    ]

    mean_angle_error = float(
        np.mean(angle_errors)
    )

    max_angle_error = float(
        np.max(angle_errors)
    )

    angle_std = float(
        np.std(angles)
    )

    # Reject polygons whose angles are too far from an octagon
    if mean_angle_error > MAX_ANGLE_ERROR:
        return 0.0, {}

    # Higher scores mean the angles are closer to ideal
    mean_angle_score = max(
        0.0,
        1.0
        - mean_angle_error / MAX_ANGLE_ERROR
    )

    # Measures how similar all 8 angles are to each other
    consistency_score = max(
        0.0,
        1.0
        - angle_std / 35.0
    )

    # Penalizes a polygon if even one angle is especially bad
    worst_angle_score = max(
        0.0,
        1.0
        - max_angle_error / 50.0
    )

    side_lengths = []

    # Calculate the distance between each pair of neighboring corners
    for i in range(8):
        p1 = np.array(
            points[i],
            dtype=np.float32
        )

        p2 = np.array(
            points[(i + 1) % 8],
            dtype=np.float32
        )

        length = float(
            np.linalg.norm(
                p2 - p1
            )
        )

        side_lengths.append(length)

    min_side = min(side_lengths)
    max_side = max(side_lengths)

    # Reject polygons whose sides are too small to analyze reliably
    if min_side < min_side_pixels:
        return 0.0, {}

    # Compare the longest side with the shortest side
    side_ratio = max_side / min_side

    if side_ratio > MAX_SIDE_LENGTH_RATIO:
        return 0.0, {}

    # Higher score means the side lengths are more similar
    side_score = max(
        0.0,
        1.0
        - (side_ratio - 1.0)
        / (MAX_SIDE_LENGTH_RATIO - 1.0)
    )

    mean_side = float(
        np.mean(side_lengths)
    )

    side_std = float(
        np.std(side_lengths)
    )

    # Coefficient of variation measures side-length consistency
    if mean_side > 0:
        side_cv = side_std / mean_side
    else:
        side_cv = 1.0

    side_consistency_score = max(
        0.0,
        1.0
        - side_cv / 0.45
    )

    # Combine all geometry measurements into one octagon score
    octagon_score = (
        0.45 * mean_angle_score
        + 0.20 * consistency_score
        + 0.10 * worst_angle_score
        + 0.15 * side_score
        + 0.10 * side_consistency_score
    )

    details = {
        "mean_angle_error": mean_angle_error,
        "max_angle_error": max_angle_error,
        "angle_std": angle_std,
        "side_ratio": side_ratio,
        "side_cv": side_cv
    }

    return octagon_score, details


def find_best_octagon(contour, min_side_pixels):
    """Simplify a contour and find its best 8-sided approximation."""

    # Convex hull removes inward dents from the contour
    hull = cv2.convexHull(contour)

    # Measure the distance around the outside boundary
    perimeter = cv2.arcLength(
        hull,
        True
    )

    if perimeter <= 0:
        return None, 0.0, {}

    best_approx = None
    best_score = 0.0
    best_details = {}

    # Different epsilon values control how strongly the contour is simplified
    epsilon_ratios = [
        0.010,
        0.0125,
        0.015,
        0.0175,
        0.020,
        0.0225,
        0.025,
        0.0275,
        0.030,
        0.0325,
        0.035,
        0.040,
        0.045
    ]

    for epsilon_ratio in epsilon_ratios:

        # Reduce the many contour points to the main polygon corners
        approx = cv2.approxPolyDP(
            hull,
            epsilon_ratio * perimeter,
            True
        )

        # 8 corner points means an 8-sided polygon
        if len(approx) != 8:
            continue

        # Check how closely those 8 corners resemble a real octagon
        score, details = score_octagon(
            approx,
            min_side_pixels
        )

        if score > best_score:
            best_score = score
            best_approx = approx
            best_details = details

    return (
        best_approx,
        best_score,
        best_details
    )


def detect_stop_sign(frame, hsv=None):
    """Detect the strongest valid red octagonal object.

    hsv optionally carries a precomputed blurred HSV frame so one blur and
    conversion can be shared between both detectors on the Pi Zero 2 W.
    """

    height, width = frame.shape[:2]
    frame_area = width * height

    if hsv is None:
        # Small blur reduces noise while preserving distant corners
        blurred = cv2.GaussianBlur(
            frame,
            (3, 3),
            0
        )

        hsv = cv2.cvtColor(
            blurred,
            cv2.COLOR_BGR2HSV
        )

    # Find all red pixels
    red_mask = get_red_mask(hsv)

    # Ignore red pixels outside the allowed frame region
    red_mask = cv2.bitwise_and(
        red_mask,
        get_roi_mask(frame)
    )

    # Find the boundaries of connected red regions
    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_detection = None
    best_score = 0.0

    # Test every red region as a possible stop sign
    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        # Normalize object size relative to the entire camera frame
        area_ratio = area / frame_area

        if not (
            STOP_MIN_AREA
            <= area_ratio
            <= STOP_MAX_AREA
        ):
            continue

        # Get a rectangular region around the red object
        x, y, box_w, box_h = cv2.boundingRect(
            contour
        )

        if box_w <= 0 or box_h <= 0:
            continue

        # Crop the candidate so its brightness can be analyzed separately
        candidate_hsv = hsv[
            y:y + box_h,
            x:x + box_w
        ]

        candidate_red_mask = red_mask[
            y:y + box_h,
            x:x + box_w
        ]

        # Reject red regions that behave more like illuminated bulbs
        if looks_like_red_light(
            candidate_hsv,
            candidate_red_mask
        ):
            continue

        # Distant signs use slightly more forgiving geometry thresholds
        if area_ratio < FAR_OBJECT_AREA_RATIO:
            min_side_pixels = FAR_MIN_SIDE_PIXELS
            min_octagon_score = FAR_MIN_OCTAGON_SCORE
            min_shape_match = FAR_MIN_SHAPE_MATCH

        else:
            min_side_pixels = NORMAL_MIN_SIDE_PIXELS
            min_octagon_score = NORMAL_MIN_OCTAGON_SCORE
            min_shape_match = NORMAL_MIN_SHAPE_MATCH

        # Compare the contour with the smallest circle that surrounds it
        _, radius = cv2.minEnclosingCircle(
            contour
        )

        if radius <= 0:
            continue

        circle_area = (
            np.pi
            * radius
            * radius
        )

        if circle_area <= 0:
            continue

        # A near-perfect circle fills its enclosing circle more than an octagon
        circle_fill_ratio = area / circle_area

        if circle_fill_ratio > MAX_CIRCLE_FILL:
            continue

        # Stop signs should be approximately as wide as they are tall
        aspect_ratio = box_w / float(box_h)

        if not (
            STOP_MIN_ASPECT
            <= aspect_ratio
            <= STOP_MAX_ASPECT
        ):
            continue

        box_area = box_w * box_h

        if box_area <= 0:
            continue

        # Extent measures how much of the bounding rectangle the object fills
        extent = area / box_area

        if extent < MIN_EXTENT:
            continue

        # Simplify the contour and search for the best 8-sided polygon
        octagon, octagon_score, details = find_best_octagon(
            contour,
            min_side_pixels
        )

        if octagon is None:
            continue

        if octagon_score < min_octagon_score:
            continue

        contour_area = cv2.contourArea(
            contour
        )

        octagon_area = cv2.contourArea(
            octagon
        )

        if contour_area <= 0:
            continue

        # Check how closely the simplified octagon covers the original contour
        shape_match_ratio = (
            octagon_area
            / contour_area
        )

        if shape_match_ratio < min_shape_match:
            continue

        # Favor objects whose width and height are similar
        square_score = max(
            0.0,
            1.0 - abs(
                1.0 - aspect_ratio
            )
        )

        extent_score = min(
            1.0,
            extent
        )

        # Geometry is weighted most heavily in the final confidence score
        final_score = (
            0.82 * octagon_score
            + 0.12 * square_score
            + 0.06 * extent_score
        )

        final_score = float(
            np.clip(
                final_score,
                0.0,
                1.0
            )
        )

        # Keep only the strongest stop-sign candidate in the frame
        if final_score > best_score:
            best_score = final_score

            (
                final_x,
                final_y,
                final_w,
                final_h
            ) = cv2.boundingRect(
                octagon
            )

            best_detection = {
                "object": "STOP_SIGN",
                "box": (
                    final_x,
                    final_y,
                    final_w,
                    final_h
                ),
                "confidence": final_score
            }

    return best_detection