import cv2
import numpy as np
# Limit detection to the useful part of the camera frame

TOP_CUTOFF = 0.00      # Start searching at the very top of the frame
BOTTOM_CUTOFF = 0.75   # Stop searching after 75% of the frame height


# Basic candidate requirements

MIN_SATURATION = 110   # Minimum color intensity; rejects gray/washed-out objects
MIN_VALUE = 170        # Minimum brightness; rejects dark colored objects

MIN_AREA_RATIO = 0.002    # Smallest allowed candidate relative to the full frame.
                          # Matches the classical backend's DETECT_MIN_AREA_FRAC so
                          # an A/B compares like with like. At 0.00020 the floor was
                          # ~15 px at 320x240 and single specular glints strobed in
                          # and out of the result every frame.
MAX_AREA_RATIO = 0.05     # Largest allowed candidate relative to the full frame

MIN_COLOR_RATIO = 0.08    # At least 8% of the candidate must contain the winning color
WINNER_RATIO = 1.20       # Winning color must be 20% stronger than the second-best color


# Red light brightness requirements

RED_MIN_MEAN_VALUE = 220      # Minimum average brightness of red pixels
RED_MIN_PEAK_VALUE = 240     # Minimum brightness of the brightest red pixels
RED_MIN_BRIGHT_RATIO = 0.40   # At least 30% of red pixels must be very bright
RED_BRIGHT_PIXEL_VALUE = 220  # Brightness required for a red pixel to count as "very bright"


# Yellow light brightness requirements

YELLOW_MIN_MEAN_VALUE = 200      # Minimum average brightness of yellow pixels
YELLOW_MIN_PEAK_VALUE = 230     # Minimum brightness of the brightest yellow pixels
YELLOW_MIN_BRIGHT_RATIO = 0.3  # At least 25% of yellow pixels must be very bright
YELLOW_BRIGHT_PIXEL_VALUE = 210  # Brightness required for a yellow pixel to count as "very bright"


# Green light brightness requirements

GREEN_MIN_MEAN_VALUE = 185      # Minimum average brightness of green pixels
GREEN_MIN_PEAK_VALUE = 220      # Minimum brightness of the brightest green pixels
GREEN_MIN_BRIGHT_RATIO = 0.25   # At least 25% of green pixels must be very bright
GREEN_BRIGHT_PIXEL_VALUE = 205  # Brightness required for a green pixel to count as "very bright"


# Final bounding-box requirements

FINAL_MIN_SATURATION = 130  # Minimum saturation used to isolate the final red/green bulb
FINAL_MIN_VALUE = 185       # Minimum brightness used to isolate the final red/green bulb

YELLOW_FINAL_MIN_SATURATION = 70  # Lower saturation allowed because yellow can appear washed out
YELLOW_FINAL_MIN_VALUE = 140       # Lower brightness allowed when isolating the yellow bulb

FINAL_BOX_PADDING = 0.10    # Add 10% padding around the final red/green bulb box
YELLOW_BOX_PADDING = 0.18   # Add 18% padding around yellow because its mask can be smaller

# HSV ranges for red, yellow, and green

RED1_LOWER = np.array([0, 100, 100])
RED1_UPPER = np.array([12, 255, 255])

RED2_LOWER = np.array([168, 100, 100])
RED2_UPPER = np.array([180, 255, 255])

YELLOW_LOWER = np.array([15, 90, 100])
YELLOW_UPPER = np.array([40, 255, 255])

GREEN_LOWER = np.array([40, 80, 80])
GREEN_UPPER = np.array([95, 255, 255])

# Kernels used to remove mask noise and reconnect nearby pixels

OPEN_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_KERNEL = np.ones((5, 5), np.uint8)
YELLOW_DILATE_KERNEL = np.ones((5, 5), np.uint8)

# Helper functions for ROI and color masks

def get_roi_mask(frame):
    """Create a mask for the allowed detection region."""

    h, w = frame.shape[:2]

    # Start with a black mask the same size as the camera frame
    mask = np.zeros((h, w), dtype=np.uint8)

    ymin = int(h * TOP_CUTOFF)
    ymax = int(h * BOTTOM_CUTOFF)

    # White pixels represent the region where detection is allowed
    mask[ymin:ymax, :] = 255

    return mask


def clean_mask(mask):
    """Remove small noise and reconnect nearby regions."""

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


def get_color_masks(hsv):
    """Create separate masks for red, yellow, and green."""

    # Red wraps around the HSV hue scale, so two red ranges are needed
    red1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    red2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    return red, yellow, green


def get_single_color_mask(hsv, color):
    """Return only the requested color mask."""

    red, yellow, green = get_color_masks(hsv)

    if color == "RED":
        return red

    if color == "YELLOW":
        return yellow

    if color == "GREEN":
        return green

    # Return an empty mask if an invalid color is requested
    return np.zeros(hsv.shape[:2], dtype=np.uint8)


def find_bright_candidates(frame, hsv=None):
    """Find bright red, yellow, or green regions that could be lights.

    hsv optionally carries a precomputed blurred HSV frame so one blur and
    conversion can be shared between both detectors on the Pi Zero 2 W.
    """

    h, w = frame.shape[:2]
    frame_area = w * h

    if hsv is None:
        # Blur slightly to reduce camera noise
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)

        # HSV separates color from brightness better than BGR
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Keep only pixels that are sufficiently saturated and bright
    intense_mask = cv2.inRange(
        hsv,
        np.array([0, MIN_SATURATION, MIN_VALUE]),
        np.array([180, 255, 255])
    )

    red, yellow, green = get_color_masks(hsv)

    # Combine all valid traffic-light colors into one mask
    color_mask = cv2.bitwise_or(red, yellow)
    color_mask = cv2.bitwise_or(color_mask, green)

    # A candidate must pass both the color and intensity tests
    candidate_mask = cv2.bitwise_and(
        intense_mask,
        color_mask
    )

    # Remove anything outside the allowed part of the frame
    candidate_mask = cv2.bitwise_and(
        candidate_mask,
        get_roi_mask(frame)
    )

    candidate_mask = clean_mask(candidate_mask)

    # Convert connected white regions into object contours
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

        # Compare candidate size to the full camera frame
        area_ratio = area / frame_area

        if not (
            MIN_AREA_RATIO
            <= area_ratio
            <= MAX_AREA_RATIO
        ):
            continue

        # Create a rectangular region around the candidate
        x, y, w, h = cv2.boundingRect(contour)

        if w <= 0 or h <= 0:
            continue

        candidates.append({
            "box": (x, y, w, h)
        })

    return candidates


def validate_light_color(hsv, color_mask, color):
    """Check whether a colored object behaves like a glowing bulb."""

    # Extract brightness values only from pixels belonging to this color
    values = hsv[:, :, 2][color_mask > 0]

    if values.size == 0:
        return False

    # Use separate brightness requirements for each bulb color
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

    # Average brightness of the detected color
    mean_value = float(np.mean(values))

    # Brightness reached by the brightest 10% of pixels
    peak_value = float(
        np.percentile(values, 90)
    )

    # Percentage of pixels bright enough to count as strongly illuminated
    bright_ratio = float(
        np.mean(values >= bright_pixel_value)
    )

    # All three brightness conditions must pass
    return (
        mean_value >= min_mean
        and peak_value >= min_peak
        and bright_ratio >= min_bright_ratio
    )


def get_precise_bulb_box(frame, original_box, color):
    """Refine the rough candidate box around the actual illuminated bulb."""

    x, y, box_w, box_h = original_box

    # Crop the full frame down to the rough candidate region
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

    # Keep only the color that already won the classification step
    color_mask = get_single_color_mask(
        hsv,
        color
    )

    # Yellow uses looser thresholds because its bright center can wash toward white
    if color == "YELLOW":
        min_saturation = YELLOW_FINAL_MIN_SATURATION
        min_value = YELLOW_FINAL_MIN_VALUE
    else:
        min_saturation = FINAL_MIN_SATURATION
        min_value = FINAL_MIN_VALUE

    strong_mask = cv2.inRange(
        hsv,
        np.array([0, min_saturation, min_value]),
        np.array([180, 255, 255])
    )

    # Pixel must be both the correct color and bright enough
    final_mask = cv2.bitwise_and(
        color_mask,
        strong_mask
    )

    # Expand yellow slightly so its final box does not become too small
    if color == "YELLOW":
        final_mask = cv2.dilate(
            final_mask,
            YELLOW_DILATE_KERNEL,
            iterations=1
        )

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

        # Create a mask containing only this contour
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

        # Measure brightness and saturation inside this contour
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

        # Favor regions that are large, bright, and strongly colored
        score = (
            area
            * mean_v
            * (0.5 + 0.5 * mean_s / 255.0)
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return original_box

    # Create the tighter box around the strongest bulb contour
    bx, by, bw, bh = cv2.boundingRect(
        best_contour
    )

    padding = (
        YELLOW_BOX_PADDING
        if color == "YELLOW"
        else FINAL_BOX_PADDING
    )

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    # Expand the box without going outside the original ROI
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

    # Convert ROI coordinates back to full-frame coordinates
    return (
        x + new_x1,
        y + new_y1,
        new_x2 - new_x1,
        new_y2 - new_y1
    )


def analyze_candidate_roi(frame, box):
    """Determine whether a candidate is red, yellow, or green."""

    x, y, box_w, box_h = box

    # Analyze only the rough candidate region
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

    # Apply the general brightness and saturation requirement again
    intense_mask = cv2.inRange(
        hsv,
        np.array([0, MIN_SATURATION, MIN_VALUE]),
        np.array([180, 255, 255])
    )

    red, yellow, green = get_color_masks(hsv)

    # Each color must also pass the intensity requirement
    masks = {
        "RED": cv2.bitwise_and(red, intense_mask),
        "YELLOW": cv2.bitwise_and(yellow, intense_mask),
        "GREEN": cv2.bitwise_and(green, intense_mask)
    }

    powers = {}
    counts = {}

    for color, mask in masks.items():

        # Count how many strong pixels belong to each color
        counts[color] = cv2.countNonZero(mask)

        values = hsv[:, :, 2][
            mask > 0
        ]

        if values.size == 0:
            powers[color] = 0.0
            continue

        # Find the brightness threshold for the brightest 15% of pixels
        threshold = np.percentile(
            values,
            85
        )

        strongest = values[
            values >= threshold
        ]

        # Score each color using brightness and number of strong pixels
        powers[color] = float(
            np.mean(strongest)
            * strongest.size
        )

    # Rank colors from strongest to weakest
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

    # Make sure enough of the candidate actually contains the winning color
    roi_area = box_w * box_h

    winner_ratio = (
        counts[winner_color]
        / roi_area
    )

    if winner_ratio < MIN_COLOR_RATIO:
        return None

    # Winning color must clearly beat the second-strongest color
    if (
        second_power > 0
        and winner_power
        < second_power * WINNER_RATIO
    ):
        return None

    # Make sure the winning color behaves like an illuminated bulb
    if not validate_light_color(
        hsv,
        masks[winner_color],
        winner_color
    ):
        return None

    # Tighten the box around the actual illuminated region
    precise_box = get_precise_bulb_box(
        frame,
        box,
        winner_color
    )

    return {
        "object": f"TRAFFIC_LIGHT_{winner_color}",
        "color": winner_color,
        "box": precise_box,
        "power": winner_power
    }


def detect_traffic_light(frame, hsv=None):
    """Run the complete traffic-light detection process.

    Returns the strongest light in this frame, or None. Flicker smoothing is
    the caller's job (the robot runs K-of-N temporal confirmation), so this
    function keeps no state between frames.
    """

    # First find rough bright-color candidate regions
    candidates = find_bright_candidates(
        frame,
        hsv
    )

    valid_lights = []

    # Analyze every candidate in more detail
    for candidate in candidates:

        result = analyze_candidate_roi(
            frame,
            candidate["box"]
        )

        if result is not None:
            valid_lights.append(
                result
            )

    # No valid traffic light was found
    if not valid_lights:
        return None

    # If multiple lights survive, choose the strongest one
    return max(
        valid_lights,
        key=lambda light: light["power"]
    )