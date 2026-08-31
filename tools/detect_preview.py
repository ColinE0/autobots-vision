"""
Live camera preview for the stop sign / traffic light detectors.

    cd autobots-vision && python -m tools.detect_preview

Two frame sources, picked by config.CAMERA_BACKEND with the same names and
values as the robot:
  'csi'  Pi camera (Arducam IMX219) via picamera2 (apt: python3-picamera2)
  'usb'  laptop / USB webcam via OpenCV, index config.CAMERA_INDEX

Runs the detectors at DETECT_PROC_WIDTH, the same size the robot processes,
so what this shows is what the robot will actually see (a detection that
only works at full webcam resolution is a detection the robot does not
have). With a display the window shows boxes, scaled 2x, q to quit. Over
SSH with no display it falls back to printing one line whenever the
detections change, plus an FPS report every 2 s. Ctrl+C to quit.
"""
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2

import config
from vision.detector_geometric import boxes_represent_same_object
from vision.stop_sign_detector import detect_stop_sign
from vision.traffic_light_detector import detect_traffic_light


def open_csi():
    from picamera2 import Picamera2
    picam = Picamera2()

    # libcamera's "RGB888" is B,G,R in memory, which is exactly OpenCV's
    # BGR. Do not "correct" it to BGR888; that one comes out RGB and every
    # hue the detectors depend on shifts. (Same note as the robot's
    # hardware/camera.py.)
    picam.configure(picam.create_video_configuration(
        main={'size': (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
              'format': 'RGB888'},
        controls={'FrameRate': float(config.CAMERA_FPS)}))

    picam.start()

    def read():
        frame = picam.capture_array('main')
        return frame is not None, frame

    return read, picam.close


def open_usb():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Camera index {config.CAMERA_INDEX} did not open. "
            "Set config.CAMERA_INDEX to your webcam's index.")

    return cap.read, cap.release


def describe(stop_sign_data, traffic_light_data):
    """One-line status for headless mode."""

    parts = []

    if stop_sign_data is not None:
        parts.append(f"STOP SIGN {stop_sign_data['confidence']:.2f}")

    if traffic_light_data is not None:
        parts.append(f"LIGHT {traffic_light_data['color']}")

    return " | ".join(parts) if parts else "(nothing)"


def main():
    if config.CAMERA_BACKEND == 'csi':
        read, close = open_csi()
    else:
        read, close = open_usb()

    headless = False
    last_line = None
    frames = 0
    fps_t0 = time.monotonic()

    try:
        while True:
            ok, frame = read()

            if not ok:
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
                    stop_sign_data = None

            frames += 1
            now = time.monotonic()

            if now - fps_t0 >= 2.0:
                print(f"{frames / (now - fps_t0):.1f} FPS")
                frames = 0
                fps_t0 = now

            if headless:
                line = describe(stop_sign_data, traffic_light_data)

                if line != last_line:
                    print(line)
                    last_line = line

                continue

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

            try:
                cv2.imshow("Subsystem Pipeline", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except cv2.error:
                # No display (SSH session): switch to console output.
                headless = True
                print("No display found, printing detections instead. "
                      "Ctrl+C to quit.")

    except KeyboardInterrupt:
        pass

    finally:
        close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
