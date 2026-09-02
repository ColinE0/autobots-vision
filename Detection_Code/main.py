import cv2

from stop_sign_detector import detect_stop_sign
from traffic_light_detector import detect_traffic_light


CAMERA_SOURCE = 0


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


def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Run detectors.
        stop_sign_data = detect_stop_sign(frame)
        traffic_light_data = detect_traffic_light(frame)

        # Traffic-light detection takes priority if both detectors
        # appear to be looking at the same physical object.
        if stop_sign_data is not None and traffic_light_data is not None:
            if boxes_represent_same_object(
                stop_sign_data["box"],
                traffic_light_data["box"]
            ):
                print(
                    "STOP SIGN SUPPRESSED: "
                    "same object as traffic light"
                )

                stop_sign_data = None

        # Draw stop sign.
        if stop_sign_data is not None:
            x, y, w, h = stop_sign_data["box"]
            confidence = stop_sign_data["confidence"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"STOP SIGN {confidence:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

        # Draw traffic light.
        if traffic_light_data is not None:
            x, y, w, h = traffic_light_data["box"]
            color = traffic_light_data["color"]

            if color == "RED":
                box_color = (0, 0, 255)

            elif color == "YELLOW":
                box_color = (0, 255, 255)

            else:
                box_color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            cv2.putText(
                frame,
                f"TRAFFIC LIGHT: {color}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

        cv2.imshow("Subsystem Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()