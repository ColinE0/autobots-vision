"""
Live camera preview for the stop sign / traffic light detectors.

    cd autobots-vision && python -m tools.detect_preview

Runs the detectors at DETECT_PROC_WIDTH, the same size the robot processes,
so what this window shows is what the robot will actually see (a detection
that only works at full webcam resolution is a detection the robot does not
have). The window is scaled 2x for readability. Press q to quit.

Camera index comes from config.CAMERA_SOURCE.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2

import config
from vision.detector_geometric import boxes_represent_same_object
from vision.stop_sign_detector import detect_stop_sign
from vision.traffic_light_detector import detect_traffic_light


def main():
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Downscale to the robot's processing width before detecting.
        h, w = frame.shape[:2]

        if w != config.DETECT_PROC_WIDTH:
            scale = config.DETECT_PROC_WIDTH / float(w)
            frame = cv2.resize(
                frame,
                (config.DETECT_PROC_WIDTH, max(1, int(h * scale)))
            )

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

        # 2x upscale so the robot-sized frame is comfortable to watch.
        fh, fw = frame.shape[:2]
        display = cv2.resize(
            frame,
            (fw * 2, fh * 2),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow("Subsystem Pipeline", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
