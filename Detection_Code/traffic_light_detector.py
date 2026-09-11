import cv2
import numpy as np
import time


# ============================================================
# Region of Interest
# ============================================================

# Only analyze the upper portion of the frame where traffic lights
# are expected to appear.
TOP_CUTOFF = 0.0
BOTTOM_CUTOFF = 0.75


# ============================================================
# General Candidate Requirements
# ============================================================

MIN_SATURATION = 110
MIN_VALUE = 170

MIN_AREA_RATIO = 0.00008
MAX_AREA_RATIO = 0.05

# Minimum percentage of the candidate ROI that must belong
# to the winning traffic-light color.
MIN_COLOR_RATIO = 0.08

# Winning color must contain at least this many times as many
# pixels as the second strongest color.
WINNER_RATIO = 1.20

# Prevent unnecessary repeated full analysis of the same detection.
HOLD_TIME = 0.25


# ============================================================
# Red-Light Thresholds
# ============================================================

RED_CANDIDATE_MIN_SATURATION = 75
RED_CANDIDATE_MIN_VALUE = 130

RED_MIN_MEAN_VALUE = 150
RED_MIN_PEAK_VALUE = 180
RED_MIN_BRIGHT_RATIO = 0.40
RED_BRIGHT_PIXEL_VALUE = 155


# ============================================================
# Yellow-Light Thresholds
# ============================================================

YELLOW_CANDIDATE_MIN_SATURATION = 65
YELLOW_CANDIDATE_MIN_VALUE = 150

YELLOW_MIN_MEAN_VALUE = 175
YELLOW_MIN_PEAK_VALUE = 220
YELLOW_MIN_BRIGHT_RATIO = 0.20
YELLOW_BRIGHT_PIXEL_VALUE = 205

# Yellow LEDs can contain a white, overexposed center.
YELLOW_WHITE_MIN_VALUE = 230
YELLOW_WHITE_MAX_SATURATION = 90
YELLOW_MIN_WHITE_CORE_RATIO = 0.04

YELLOW_MIN_HOT_RATIO = 0.25
YELLOW_MIN_SUPERHOT_RATIO = 0.04
YELLOW_HOT_PIXEL_VALUE = 235
YELLOW_SUPERHOT_PIXEL_VALUE = 248


# ============================================================
# Green-Light Thresholds
# ============================================================

GREEN_MIN_MEAN_VALUE = 185
GREEN_MIN_PEAK_VALUE = 215
GREEN_MIN_BRIGHT_RATIO = 0.25
GREEN_BRIGHT_PIXEL_VALUE = 205


# ============================================================
# Final Bulb-Box Requirements
# ============================================================

FINAL_MIN_SATURATION = 130
FINAL_MIN_VALUE = 185

YELLOW_FINAL_MIN_SATURATION = 70
YELLOW_FINAL_MIN_VALUE = 140

FINAL_BOX_PADDING = 0.10
YELLOW_BOX_PADDING = 0.18


# ============================================================
# HSV Color Ranges
# ============================================================

# Red wraps around the HSV hue scale, so two ranges are required.
RED1_LOWER = np.array([0, 100, 100])
RED1_UPPER = np.array([8, 255, 255])

RED2_LOWER = np.array([168, 100, 100])
RED2_UPPER = np.array([180, 255, 255])

YELLOW_LOWER = np.array([9, 90, 100])
YELLOW_UPPER = np.array([40, 255, 255])

GREEN_LOWER = np.array([40, 80, 80])
GREEN_UPPER = np.array([95, 255, 255])


# ============================================================
# Morphology Kernels
# ============================================================

OPEN_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_KERNEL = np.ones((5, 5), np.uint8)

YELLOW_DILATE_KERNEL = np.ones((5, 5), np.uint8)
YELLOW_WHITE_NEIGHBOR_KERNEL = np.ones((7, 7), np.uint8)


# ============================================================
# Detection Hold State
# ============================================================

last_detection = None
last_analysis_time = 0.0


# ============================================================
# Region of Interest
# ============================================================

def get_roi_mask(frame):
    """
    Create a mask limiting traffic-light detection to the allowed
    vertical portion of the image.
    """

    height, width = frame.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    ymin = int(height * TOP_CUTOFF)
    ymax = int(height * BOTTOM_CUTOFF)

    mask[ymin:ymax, :] = 255

    return mask


# ============================================================
# Mask Cleanup
# ============================================================

def clean_mask(mask):
    """
    Remove isolated noise and reconnect nearby pixels belonging
    to the same object.
    """

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        OPEN_KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        CLOSE_KERNEL
    )

    return mask


# ============================================================
# HSV Color Masks
# ============================================================

def get_color_masks(hsv):
    """Return binary masks for red, yellow, and green pixels."""

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

    red = cv2.bitwise_or(
        red1,
        red2
    )

    yellow = cv2.inRange(
        hsv,
        YELLOW_LOWER,
        YELLOW_UPPER
    )

    green = cv2.inRange(
        hsv,
        GREEN_LOWER,
        GREEN_UPPER
    )

    return red, yellow, green


def get_single_color_mask(hsv, color):
    """Return the HSV mask corresponding to one requested color."""

    red, yellow, green = get_color_masks(hsv)

    if color == "RED":
        return red

    if color == "YELLOW":
        return yellow

    if color == "GREEN":
        return green

    return np.zeros(
        hsv.shape[:2],
        dtype=np.uint8
    )


# ============================================================
# Yellow White-Core Handling
# ============================================================

def get_yellow_with_white_core(hsv, yellow_mask):
    """
    Include white-hot pixels located next to yellow pixels.

    Bright yellow LEDs may become partially white in the camera
    because of overexposure.
    """

    white_core = cv2.inRange(
        hsv,
        np.array([
            0,
            0,
            YELLOW_WHITE_MIN_VALUE
        ]),
        np.array([
            180,
            YELLOW_WHITE_MAX_SATURATION,
            255
        ])
    )

    yellow_neighborhood = cv2.dilate(
        yellow_mask,
        YELLOW_WHITE_NEIGHBOR_KERNEL,
        iterations=1
    )

    white_near_yellow = cv2.bitwise_and(
        white_core,
        yellow_neighborhood
    )

    yellow_with_core = cv2.bitwise_or(
        yellow_mask,
        white_near_yellow
    )

    return yellow_with_core, white_near_yellow


# ============================================================
# Candidate Detection
# ============================================================

def find_bright_candidates(frame):
    """
    Find bright red, yellow, or green regions that could represent
    traffic-light bulbs.
    """

    height, width = frame.shape[:2]

    frame_area = width * height

    blurred = cv2.GaussianBlur(
        frame,
        (3, 3),
        0
    )

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # General bright and saturated mask.
    intense_mask = cv2.inRange(
        hsv,
        np.array([
            0,
            MIN_SATURATION,
            MIN_VALUE
        ]),
        np.array([
            180,
            255,
            255
        ])
    )

    red, yellow, green = get_color_masks(hsv)

    # --------------------------------------------------------
    # Red candidate mask
    # --------------------------------------------------------

    red_gate = cv2.inRange(
        hsv,
        np.array([
            0,
            RED_CANDIDATE_MIN_SATURATION,
            RED_CANDIDATE_MIN_VALUE
        ]),
        np.array([
            180,
            255,
            255
        ])
    )

    red_candidate = cv2.bitwise_and(
        red,
        red_gate
    )

    # --------------------------------------------------------
    # Green candidate mask
    # --------------------------------------------------------

    green_candidate = cv2.bitwise_and(
        green,
        intense_mask
    )

    # --------------------------------------------------------
    # Yellow candidate mask
    # --------------------------------------------------------

    yellow_gate = cv2.inRange(
        hsv,
        np.array([
            0,
            YELLOW_CANDIDATE_MIN_SATURATION,
            YELLOW_CANDIDATE_MIN_VALUE
        ]),
        np.array([
            180,
            255,
            255
        ])
    )

    yellow_candidate = cv2.bitwise_and(
        yellow,
        yellow_gate
    )

    _, yellow_white_core = get_yellow_with_white_core(
        hsv,
        yellow
    )

    yellow_candidate = cv2.bitwise_or(
        yellow_candidate,
        yellow_white_core
    )

    # Combine all possible traffic-light candidates.
    candidate_mask = cv2.bitwise_or(
        red_candidate,
        yellow_candidate
    )

    candidate_mask = cv2.bitwise_or(
        candidate_mask,
        green_candidate
    )

    # Apply vertical ROI.
    candidate_mask = cv2.bitwise_and(
        candidate_mask,
        get_roi_mask(frame)
    )

    candidate_mask = clean_mask(
        candidate_mask
    )

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        area_ratio = (
            area / frame_area
        )

        if not (
            MIN_AREA_RATIO
            <= area_ratio
            <= MAX_AREA_RATIO
        ):
            continue

        x, y, box_width, box_height = (
            cv2.boundingRect(
                contour
            )
        )

        if box_width <= 0 or box_height <= 0:
            continue

        candidates.append(
            (
                x,
                y,
                box_width,
                box_height
            )
        )

    return candidates


# ============================================================
# Brightness Validation
# ============================================================

def validate_light_color(hsv, color_mask, color):
    """
    Verify that the selected color is bright enough to represent
    an illuminated traffic light.
    """

    values = hsv[:, :, 2][
        color_mask > 0
    ]

    if values.size == 0:
        return False

    if color == "RED":

        min_mean = RED_MIN_MEAN_VALUE
        min_peak = RED_MIN_PEAK_VALUE
        min_bright_ratio = RED_MIN_BRIGHT_RATIO
        bright_pixel_value = RED_BRIGHT_PIXEL_VALUE

    elif color == "YELLOW":

        min_mean = YELLOW_MIN_MEAN_VALUE
        min_peak = YELLOW_MIN_PEAK_VALUE
        min_bright_ratio = YELLOW_MIN_BRIGHT_RATIO
        bright_pixel_value = YELLOW_BRIGHT_PIXEL_VALUE

    elif color == "GREEN":

        min_mean = GREEN_MIN_MEAN_VALUE
        min_peak = GREEN_MIN_PEAK_VALUE
        min_bright_ratio = GREEN_MIN_BRIGHT_RATIO
        bright_pixel_value = GREEN_BRIGHT_PIXEL_VALUE

    else:
        return False

    mean_value = float(
        np.mean(values)
    )

    peak_value = float(
        np.percentile(
            values,
            90
        )
    )

    bright_ratio = float(
        np.mean(
            values >= bright_pixel_value
        )
    )

    return (
        mean_value >= min_mean
        and peak_value >= min_peak
        and bright_ratio >= min_bright_ratio
    )


# ============================================================
# Precise Bulb Bounding Box
# ============================================================

def get_precise_bulb_box(frame, original_box, color):
    """
    Refine a candidate bounding box so that it surrounds the
    illuminated bulb rather than the entire candidate region.
    """

    x, y, box_width, box_height = original_box

    roi = frame[
        y:y + box_height,
        x:x + box_width
    ]

    if roi is None or roi.size == 0:
        return original_box

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    color_mask = get_single_color_mask(
        hsv,
        color
    )

    if color == "YELLOW":

        min_saturation = (
            YELLOW_FINAL_MIN_SATURATION
        )

        min_value = (
            YELLOW_FINAL_MIN_VALUE
        )

    else:

        min_saturation = (
            FINAL_MIN_SATURATION
        )

        min_value = (
            FINAL_MIN_VALUE
        )

    strong_mask = cv2.inRange(
        hsv,
        np.array([
            0,
            min_saturation,
            min_value
        ]),
        np.array([
            180,
            255,
            255
        ])
    )

    final_mask = cv2.bitwise_and(
        color_mask,
        strong_mask
    )

    # Color-specific cleanup.
    if color == "YELLOW":

        final_mask = cv2.dilate(
            final_mask,
            YELLOW_DILATE_KERNEL,
            iterations=1
        )

    elif color == "RED":

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            CLOSE_KERNEL
        )

    else:

        final_mask = clean_mask(
            final_mask
        )

    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return original_box

    best_contour = None
    best_score = 0.0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        contour_mask = np.zeros(
            final_mask.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            cv2.FILLED
        )

        values_v = hsv[:, :, 2][
            contour_mask > 0
        ]

        values_s = hsv[:, :, 1][
            contour_mask > 0
        ]

        if (
            values_v.size == 0
            or values_s.size == 0
        ):
            continue

        mean_v = float(
            np.mean(values_v)
        )

        mean_s = float(
            np.mean(values_s)
        )

        # Favor regions that are larger, brighter, and more saturated.
        score = (
            area
            * mean_v
            * (
                0.5
                + 0.5
                * mean_s
                / 255.0
            )
        )

        if score > best_score:

            best_score = score
            best_contour = contour

    if best_contour is None:
        return original_box

    bx, by, bw, bh = cv2.boundingRect(
        best_contour
    )

    if color == "YELLOW":
        padding = YELLOW_BOX_PADDING

    else:
        padding = FINAL_BOX_PADDING

    pad_x = int(
        bw * padding
    )

    pad_y = int(
        bh * padding
    )

    new_x1 = max(
        0,
        bx - pad_x
    )

    new_y1 = max(
        0,
        by - pad_y
    )

    new_x2 = min(
        box_width,
        bx + bw + pad_x
    )

    new_y2 = min(
        box_height,
        by + bh + pad_y
    )

    return (
        x + new_x1,
        y + new_y1,
        new_x2 - new_x1,
        new_y2 - new_y1
    )


# ============================================================
# Candidate Analysis
# ============================================================

def analyze_candidate_roi(frame, box):
    """
    Analyze one candidate region.

    Processing order:
    1. Determine the dominant traffic-light color.
    2. Lock that color.
    3. Validate its brightness.
    4. Refine the bulb bounding box.
    """

    x, y, box_width, box_height = box

    roi = frame[
        y:y + box_height,
        x:x + box_width
    ]

    if roi is None or roi.size == 0:
        return None

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    red_mask, yellow_mask, green_mask = (
        get_color_masks(hsv)
    )

    masks = {
        "RED": red_mask,
        "YELLOW": yellow_mask,
        "GREEN": green_mask
    }

    counts = {
        color: cv2.countNonZero(mask)
        for color, mask in masks.items()
    }

    roi_area = max(
        box_width * box_height,
        1
    )

    # --------------------------------------------------------
    # Stage 1: Color classification
    # --------------------------------------------------------

    sorted_colors = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    winner_color = sorted_colors[0][0]
    winner_count = sorted_colors[0][1]

    second_count = sorted_colors[1][1]

    if winner_count <= 0:
        return None

    winner_ratio = (
        winner_count / roi_area
    )

    # Candidate must contain enough pixels of the winning color.
    if winner_ratio < MIN_COLOR_RATIO:
        return None

    # Winning color must clearly beat the second strongest color.
    if (
        second_count > 0
        and winner_count
        < second_count * WINNER_RATIO
    ):
        return None

    # Once selected here, brightness is not allowed to change color.
    winner_mask = masks[
        winner_color
    ]

    # --------------------------------------------------------
    # Stage 2: Brightness validation
    # --------------------------------------------------------

    if not validate_light_color(
        hsv,
        winner_mask,
        winner_color
    ):
        return None

    # --------------------------------------------------------
    # Detection strength
    # --------------------------------------------------------

    values = hsv[:, :, 2][
        winner_mask > 0
    ]

    if values.size > 0:

        strong_threshold = np.percentile(
            values,
            85
        )

        strongest = values[
            values >= strong_threshold
        ]

        mean_strong = float(
            np.mean(strongest)
        )

        peak = float(
            np.percentile(
                values,
                95
            )
        )

        winner_power = (
            0.65 * mean_strong
            + 0.35 * peak
        )

    else:

        winner_power = 0.0

    glow_score = winner_power

    # --------------------------------------------------------
    # Final bulb location
    # --------------------------------------------------------

    precise_box = get_precise_bulb_box(
        frame,
        box,
        winner_color
    )

    px, py, pw, ph = precise_box

    center_x = (
        px + pw // 2
    )

    center_y = (
        py + ph // 2
    )

    return {
        "object": f"TRAFFIC_LIGHT_{winner_color}",
        "color": winner_color,
        "box": precise_box,
        "center": (
            center_x,
            center_y
        ),
        "power": winner_power,
        "glow_score": glow_score
    }


# ============================================================
# Main Traffic-Light Detector
# ============================================================

def detect_traffic_light(frame):
    """
    Detect the strongest valid traffic light in the frame.

    Returns a detection dictionary for RED, YELLOW, or GREEN,
    or None when no valid traffic light is found.
    """

    global last_detection
    global last_analysis_time

    current_time = time.time()

    # Temporarily reuse the most recent detection.
    if (
        last_detection is not None
        and current_time - last_analysis_time
        < HOLD_TIME
    ):
        return last_detection

    candidates = find_bright_candidates(
        frame
    )

    valid_lights = []

    for box in candidates:

        result = analyze_candidate_roi(
            frame,
            box
        )

        if result is not None:
            valid_lights.append(
                result
            )

    if not valid_lights:

        last_detection = None
        last_analysis_time = current_time

        return None

    # If multiple valid lights exist, choose the strongest illuminated bulb.
    best_light = max(
        valid_lights,
        key=lambda light: light.get(
            "glow_score",
            light["power"]
        )
    )

    last_detection = best_light
    last_analysis_time = current_time

    return best_light
