"""
Traffic Light Detector (OpenCV)

High-level pipeline:
1. Blur the frame slightly and convert it to HSV.
2. Build separate red, yellow, and green color masks.
3. Keep only bright colored regions inside the allowed detection area.
4. Treat yellow differently because real yellow LEDs often wash toward white.
5. Validate each candidate using brightness statistics.
6. Refine the bounding box around the strongest bulb-shaped color region.
7. Rank valid candidates and return the light that behaves most like a real glowing bulb.
8. Briefly hold the last detection to reduce flicker between frames.

Important yellow behavior:
- Solid yellow objects used to win because they had many yellow pixels.
- The detector now avoids rewarding raw pixel count.
- Yellow LEDs are allowed to include a white-hot center.
- Yellow candidates are ranked using glow distribution so a real bulb can beat
  small yellow buttons or glossy reflections.
"""

import cv2
import numpy as np
import time


# Detection region
# Only search the upper portion of the image where traffic lights are expected.
# This reduces false positives from the floor / robot body / nearby objects.
TOP_CUTOFF = 0.00
BOTTOM_CUTOFF = 0.75


# Basic candidate requirements
# These thresholds control which bright colored blobs are large enough and
# strong enough to be considered possible traffic lights.
MIN_SATURATION = 110
MIN_VALUE = 170

MIN_AREA_RATIO = 0.00020
MAX_AREA_RATIO = 0.05

MIN_COLOR_RATIO = 0.08
WINNER_RATIO = 1.20

HOLD_TIME = 0.25


# Red light brightness requirements
RED_MIN_MEAN_VALUE = 215
RED_MIN_PEAK_VALUE = 240
RED_MIN_BRIGHT_RATIO = 0.40
RED_BRIGHT_PIXEL_VALUE = 220


# Yellow light brightness requirements
# Yellow is intentionally looser than red/green because a bright yellow LED
# often loses saturation and becomes pale or nearly white in the camera.
YELLOW_MIN_MEAN_VALUE = 175
YELLOW_MIN_PEAK_VALUE = 220
YELLOW_MIN_BRIGHT_RATIO = 0.18
YELLOW_BRIGHT_PIXEL_VALUE = 195

# Yellow gets its own looser saturation gate because a real LED often
# washes toward white in the center.
YELLOW_CANDIDATE_MIN_SATURATION = 65
YELLOW_CANDIDATE_MIN_VALUE = 150

# Real yellow LEDs can wash toward white in the center.
# Treat a very bright, low-saturation region touching yellow as
# part of the yellow bulb, but require enough of it to reject
# ordinary solid-yellow objects.
YELLOW_WHITE_MIN_VALUE = 230
YELLOW_WHITE_MAX_SATURATION = 90
YELLOW_MIN_WHITE_CORE_RATIO = 0.01

# Extra yellow glow metrics
# These are used for ranking, not just hard rejection.
# A real illuminated bulb should contain a broader region of very bright pixels.
# A yellow button may have one shiny highlight, but usually not a distributed glow.
YELLOW_HOT_PIXEL_VALUE = 235
YELLOW_SUPERHOT_PIXEL_VALUE = 248


# Green light brightness requirements
GREEN_MIN_MEAN_VALUE = 185
GREEN_MIN_PEAK_VALUE = 220
GREEN_MIN_BRIGHT_RATIO = 0.25
GREEN_BRIGHT_PIXEL_VALUE = 205


# Final bulb-box requirements
FINAL_MIN_SATURATION = 130
FINAL_MIN_VALUE = 185

YELLOW_FINAL_MIN_SATURATION = 70
YELLOW_FINAL_MIN_VALUE = 140

FINAL_BOX_PADDING = 0.10
YELLOW_BOX_PADDING = 0.18


# HSV color ranges
RED1_LOWER = np.array([0, 100, 100])
RED1_UPPER = np.array([12, 255, 255])

RED2_LOWER = np.array([168, 100, 100])
RED2_UPPER = np.array([180, 255, 255])

YELLOW_LOWER = np.array([15, 90, 100])
YELLOW_UPPER = np.array([40, 255, 255])

GREEN_LOWER = np.array([40, 80, 80])
GREEN_UPPER = np.array([95, 255, 255])


# Mask-cleaning kernels
OPEN_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_KERNEL = np.ones((5, 5), np.uint8)
YELLOW_DILATE_KERNEL = np.ones((5, 5), np.uint8)
YELLOW_WHITE_NEIGHBOR_KERNEL = np.ones((7, 7), np.uint8)


# Detection memory used to reduce flickering
# When a valid light is found, keep returning it briefly instead of immediately
# dropping the detection if one frame is noisy.
last_detection = None
last_analysis_time = 0.0


def get_roi_mask(frame):
    """Create the allowed traffic-light detection region."""

    h, w = frame.shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    ymin = int(h * TOP_CUTOFF)
    ymax = int(h * BOTTOM_CUTOFF)

    mask[ymin:ymax, :] = 255

    return mask


def clean_mask(mask):
    """Remove small noise and reconnect nearby regions."""

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


def get_color_masks(hsv):
    """Create separate red, yellow, and green HSV masks.

    Red uses two hue ranges because HSV red wraps around the hue scale:
    one range is near 0 degrees and the other is near 180.
    """

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
    """Return the mask for one traffic-light color."""

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


def get_yellow_with_white_core(hsv, yellow_mask):
    """Add white-hot pixels that are directly connected to yellow.

    Very bright yellow LEDs can become low-saturation/white in the center.
    These pixels are only accepted if they are touching a yellow region,
    which helps avoid treating unrelated white objects as part of the bulb.
    """

    white_core = cv2.inRange(
        hsv,
        np.array([0, 0, YELLOW_WHITE_MIN_VALUE]),
        np.array([180, YELLOW_WHITE_MAX_SATURATION, 255])
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


def find_bright_candidates(frame):
    """Find bright colored regions that could be traffic lights.

    Red and green use the normal global saturation/brightness gate.
    Yellow uses a separate, looser gate so the washed-out center of a real
    yellow LED is not accidentally removed.
    """

    h, w = frame.shape[:2]
    frame_area = w * h

    blurred = cv2.GaussianBlur(
        frame,
        (3, 3),
        0
    )

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    intense_mask = cv2.inRange(
        hsv,
        np.array([0, MIN_SATURATION, MIN_VALUE]),
        np.array([180, 255, 255])
    )

    red, yellow, green = get_color_masks(hsv)

    yellow_with_core, _ = get_yellow_with_white_core(
        hsv,
        yellow
    )

    # Red/green keep the normal strong saturation+brightness requirement.
    red_candidate = cv2.bitwise_and(red, intense_mask)
    green_candidate = cv2.bitwise_and(green, intense_mask)

    # Yellow needs a separate gate. A glowing yellow LED can become pale/white
    # in the center, so demanding the global saturation threshold can erase it.
    yellow_gate = cv2.inRange(
        hsv,
        np.array([0, YELLOW_CANDIDATE_MIN_SATURATION, YELLOW_CANDIDATE_MIN_VALUE]),
        np.array([180, 255, 255])
    )
    yellow_candidate = cv2.bitwise_and(yellow, yellow_gate)

    _, yellow_white_core = get_yellow_with_white_core(hsv, yellow)
    yellow_candidate = cv2.bitwise_or(yellow_candidate, yellow_white_core)

    candidate_mask = cv2.bitwise_or(red_candidate, yellow_candidate)
    candidate_mask = cv2.bitwise_or(candidate_mask, green_candidate)

    candidate_mask = cv2.bitwise_and(
        candidate_mask,
        get_roi_mask(frame)
    )

    candidate_mask = clean_mask(candidate_mask)

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        area_ratio = area / frame_area

        if not (
            MIN_AREA_RATIO
            <= area_ratio
            <= MAX_AREA_RATIO
        ):
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)

        if box_w <= 0 or box_h <= 0:
            continue

        candidates.append(
            (x, y, box_w, box_h)
        )

    return candidates


def validate_light_color(hsv, color_mask, color):
    """Return True when the color region is bright enough to behave like a lit bulb.

    Validation uses:
    - mean brightness
    - high-percentile brightness
    - fraction of pixels above a bright threshold

    Requiring all three helps reject colored surfaces that match the hue
    but are not actually illuminated.
    """

    values = hsv[:, :, 2][color_mask > 0]

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

    mean_value = float(np.mean(values))
    peak_value = float(np.percentile(values, 90))
    bright_ratio = float(
        np.mean(values >= bright_pixel_value)
    )

    return (
        mean_value >= min_mean
        and peak_value >= min_peak
        and bright_ratio >= min_bright_ratio
    )


def get_precise_bulb_box(frame, original_box, color):
    """Refine the rough candidate box around the illuminated bulb.

    The original contour can include extra nearby pixels. This function rebuilds
    a stronger color/brightness mask inside the candidate and chooses the best
    internal contour to produce a tighter bulb bounding box.
    """

    x, y, box_w, box_h = original_box

    roi = frame[
        y:y + box_h,
        x:x + box_w
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
        min_saturation = YELLOW_FINAL_MIN_SATURATION
        min_value = YELLOW_FINAL_MIN_VALUE
    else:
        min_saturation = FINAL_MIN_SATURATION
        min_value = FINAL_MIN_VALUE

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

    if color == "YELLOW":
        final_mask = cv2.dilate(
            final_mask,
            YELLOW_DILATE_KERNEL,
            iterations=1
        )

    if color == "RED":
        # Preserve small distant red-light regions.
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            CLOSE_KERNEL
        )
    else:
        final_mask = clean_mask(final_mask)

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
        area = cv2.contourArea(contour)

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

        mean_v = float(np.mean(values_v))
        mean_s = float(np.mean(values_s))

        score = (
            area
            * mean_v
            * (
                0.5
                + 0.5 * mean_s / 255.0
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

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    new_x1 = max(0, bx - pad_x)
    new_y1 = max(0, by - pad_y)

    new_x2 = min(
        box_w,
        bx + bw + pad_x
    )

    new_y2 = min(
        box_h,
        by + bh + pad_y
    )

    return (
        x + new_x1,
        y + new_y1,
        new_x2 - new_x1,
        new_y2 - new_y1
    )


def analyze_candidate_roi(frame, box):
    """Classify one candidate as red, yellow, or green.

    Each color gets a score based mainly on the brightest part of the region.
    Raw pixel count is intentionally NOT multiplied into the score because that
    previously let large solid-colored objects beat small glowing LEDs.
    """

    x, y, box_w, box_h = box

    roi = frame[
        y:y + box_h,
        x:x + box_w
    ]

    if roi is None or roi.size == 0:
        return None

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    intense_mask = cv2.inRange(
        hsv,
        np.array([0, MIN_SATURATION, MIN_VALUE]),
        np.array([180, 255, 255])
    )

    red, yellow, green = get_color_masks(hsv)

    yellow_with_core, yellow_white_core = (
        get_yellow_with_white_core(
            hsv,
            yellow
        )
    )

    yellow_gate = cv2.inRange(
        hsv,
        np.array([0, YELLOW_CANDIDATE_MIN_SATURATION, YELLOW_CANDIDATE_MIN_VALUE]),
        np.array([180, 255, 255])
    )
    yellow_bright = cv2.bitwise_and(yellow, yellow_gate)
    yellow_glow_mask = cv2.bitwise_or(yellow_bright, yellow_white_core)

    masks = {
        "RED": cv2.bitwise_and(red, intense_mask),
        "YELLOW": yellow_glow_mask,
        "GREEN": cv2.bitwise_and(green, intense_mask)
    }

    powers = {}
    counts = {}

    for color, mask in masks.items():
        counts[color] = cv2.countNonZero(mask)
        values = hsv[:, :, 2][mask > 0]

        if values.size == 0:
            powers[color] = 0.0
            continue

        threshold = np.percentile(values, 85)
        strongest = values[values >= threshold]

        # Do NOT multiply by pixel count.
        # Earlier versions did this and a large solid-yellow object could beat
        # the actual traffic light simply because it covered more pixels.
        mean_strong = float(np.mean(strongest))
        peak = float(np.percentile(values, 95))
        powers[color] = 0.65 * mean_strong + 0.35 * peak

        if color == "YELLOW":
            # A real illuminated yellow bulb should have MANY very bright pixels,
            # not just one shiny reflection. These ratios measure how much of the
            # detected yellow region behaves like a real light source.
            roi_area_safe = max(box_w * box_h, 1)
            core_ratio = cv2.countNonZero(yellow_white_core) / roi_area_safe

            yellow_values = hsv[:, :, 2][yellow_glow_mask > 0]
            if yellow_values.size > 0:
                hot_ratio = float(np.mean(yellow_values >= YELLOW_HOT_PIXEL_VALUE))
                superhot_ratio = float(np.mean(yellow_values >= YELLOW_SUPERHOT_PIXEL_VALUE))
            else:
                hot_ratio = 0.0
                superhot_ratio = 0.0

            # Reward distributed glow.
            # This is what helps the real yellow bulb outrank yellow buttons,
            # paint, plastic, or small reflective highlights.
            powers[color] += (90.0 * core_ratio)
            powers[color] += (45.0 * hot_ratio)
            powers[color] += (35.0 * superhot_ratio)

    sorted_colors = sorted(
        powers.items(),
        key=lambda item: item[1],
        reverse=True
    )

    winner_color = sorted_colors[0][0]
    winner_power = sorted_colors[0][1]
    second_power = sorted_colors[1][1]

    if winner_power <= 0:
        return None

    roi_area = box_w * box_h
    winner_ratio = counts[winner_color] / roi_area

    if winner_ratio < MIN_COLOR_RATIO:
        return None

    if (
        second_power > 0
        and winner_power < second_power * WINNER_RATIO
    ):
        return None

    if winner_color == "YELLOW":
        white_core_ratio = (
            cv2.countNonZero(yellow_white_core)
            / roi_area
        )

        if white_core_ratio < YELLOW_MIN_WHITE_CORE_RATIO:
            return None

    if not validate_light_color(
        hsv,
        masks[winner_color],
        winner_color
    ):
        return None

    # Save an explicit glow score for final candidate selection.
    # This is important for when a yellow button and the real bulb
    # are both technically valid in the same frame.
    glow_score = winner_power

    # Extra final yellow glow ranking
    # This second glow score is used when multiple valid candidates are present.
    # It gives the actual illuminated bulb another advantage over small objects
    # that technically passed the earlier yellow thresholds.
    
    if winner_color == "YELLOW":
        yellow_values = hsv[:, :, 2][masks["YELLOW"] > 0]
        if yellow_values.size > 0:
            hot_ratio = float(np.mean(yellow_values >= YELLOW_HOT_PIXEL_VALUE))
            superhot_ratio = float(np.mean(yellow_values >= YELLOW_SUPERHOT_PIXEL_VALUE))
        else:
            hot_ratio = 0.0
            superhot_ratio = 0.0

        white_core_ratio = cv2.countNonZero(yellow_white_core) / max(roi_area, 1)

        # Final yellow ranking heavily favors distributed illumination.
        glow_score += 120.0 * white_core_ratio
        glow_score += 60.0 * hot_ratio
        glow_score += 45.0 * superhot_ratio

    precise_box = get_precise_bulb_box(
        frame,
        box,
        winner_color
    )

    x, y, box_w, box_h = precise_box

    center_x = x + box_w // 2
    center_y = y + box_h // 2

    return {
        "object": f"TRAFFIC_LIGHT_{winner_color}",
        "color": winner_color,
        "box": precise_box,
        "center": (center_x, center_y),
        "power": winner_power,
        "glow_score": glow_score
    }


def detect_traffic_light(frame):
    """Detect and return the strongest valid traffic light in the frame.

    Candidate regions are analyzed independently, then the result with the best
    glow score is selected. Yellow receives extra glow-based ranking because it
    is the color most likely to be confused with solid objects.
    """

    global last_detection
    global last_analysis_time

    current_time = time.time()

    if (
        last_detection is not None
        and current_time - last_analysis_time < HOLD_TIME
    ):
        return last_detection

    candidates = find_bright_candidates(frame)

    valid_lights = []

    for box in candidates:
        result = analyze_candidate_roi(
            frame,
            box
        )

        if result is not None:
            valid_lights.append(result)

    if not valid_lights:
        last_detection = None
        last_analysis_time = current_time
        return None

    # Pick the candidate that behaves most like an actual illuminated bulb.
    # For yellow this favors a broad hot/white core instead of a shiny button.
    best_light = max(
        valid_lights,
        key=lambda light: light.get("glow_score", light["power"])
    )

    last_detection = best_light
    last_analysis_time = current_time

    color = best_light["color"]
    center_x, center_y = best_light["center"]

    print(
        f"{color} TRAFFIC LIGHT DETECTED | "
        f"Center: ({center_x}, {center_y})"
    )

    return best_light