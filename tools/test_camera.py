"""
Camera + detector smoke test. Prints FPS and any detections. No display
needed (headless). Point the camera at a stop sign / colored light to verify.
On the CSI camera: no red hits on a real stop sign usually means bad color
order (see the RGB888 note in hardware/camera.py), and a soft image means
the lens barrel needs a twist (the Arducam IMX219 focuses by rotating it).

    cd autonomous-prime && python3 -m tools.test_camera
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
from hardware.camera import make_camera
from vision.detector import make_detector


def main():
    cam = make_camera(config)
    det = make_detector(config)
    print(f"camera={config.CAMERA_BACKEND}  detector={config.DETECTOR_BACKEND}. "
          f"Ctrl+C to quit.\n")
    last_id, n, t0 = -1, 0, time.monotonic()
    try:
        while True:
            fid, frame = cam.read()
            if fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            dets = det.detect(frame)
            n += 1
            dt = time.monotonic() - t0
            if dt >= 2.0:
                fps = n / dt
                labels = ', '.join(f"{d.label}({d.area_frac*100:.1f}%)" for d in dets) or '-'
                print(f"{fps:5.1f} FPS   {labels}")
                n, t0 = 0, time.monotonic()
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()


if __name__ == '__main__':
    main()
