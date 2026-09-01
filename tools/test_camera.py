"""
Camera + detector smoke test. Prints FPS and any detections. No display
needed (headless). Point the camera at a stop sign / colored light to verify.
On the CSI camera: no red hits on a real stop sign usually means bad color
order (see the RGB888 note in hardware/camera.py), and a soft image means
the lens barrel needs a twist (the Arducam IMX219 focuses by rotating it).

    python3 -m tools.test_camera

Lines are timestamped and appended to test_camera.log in the repo folder.
The session header carries the git revision plus the camera and detector
backends, so an A/B run months later is attributable to a code state.

Detections print when the SET OF LABELS changes, not on a timer, so a
hand-held prop reads as transitions rather than a 1-in-60 sample. The area
percentages shown are from the frame that triggered the change. FPS is
reported separately every 2 s.
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
from hardware.camera import make_camera
from vision.detector import make_detector
from tools.sessionlog import SessionLog


def main():
    cam = make_camera(config)
    det = make_detector(config)
    log = SessionLog('test_camera',
                     f"camera={config.CAMERA_BACKEND} "
                     f"detector={config.DETECTOR_BACKEND} "
                     f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}@{config.CAMERA_FPS}")
    print(f"camera={config.CAMERA_BACKEND}  detector={config.DETECTOR_BACKEND}. "
          f"Ctrl+C to quit. Logging to {log.path}\n")
    last_id, n, t0 = -1, 0, time.monotonic()
    last_key = None
    try:
        while True:
            fid, frame = cam.read()
            if fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            dets = det.detect(frame)
            n += 1
            # Key on the labels alone. area_frac jitters every frame, so
            # keying on the printed string would emit at the frame rate.
            key = tuple(d.label for d in dets)
            if key != last_key:
                labels = ', '.join(
                    f"{d.label}({d.area_frac*100:.1f}%)" for d in dets) or '-'
                print(log.line(labels, echo=False), flush=True)
                last_key = key
            dt = time.monotonic() - t0
            if dt >= 2.0:
                print(log.line(f"{n / dt:5.1f} FPS", echo=False), flush=True)
                n, t0 = 0, time.monotonic()
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()
        log.close()


if __name__ == '__main__':
    main()
