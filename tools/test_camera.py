"""
Camera + detector smoke test. Prints FPS and any detections. No display
needed (headless). Point the camera at a stop sign / colored light to verify.
On the CSI camera: no red hits on a real stop sign usually means bad color
order (see the RGB888 note in hardware/camera.py), and a soft image means
the lens barrel needs a twist (the Arducam IMX219 focuses by rotating it).

    python3 -m tools.test_camera

Lines are timestamped and appended to test_camera.log in the repo folder
(with the active detector backend in the session header), so A/B runs
leave a record.
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
from hardware.camera import make_camera
from vision.detector import make_detector


def main():
    cam = make_camera(config)
    det = make_detector(config)
    log_path = pathlib.Path(__file__).resolve().parents[1] / 'test_camera.log'
    log = open(log_path, 'a')
    log.write(time.strftime('--- session %Y-%m-%d %H:%M:%S  ')
              + f"camera={config.CAMERA_BACKEND} detector={config.DETECTOR_BACKEND}\n")
    print(f"camera={config.CAMERA_BACKEND}  detector={config.DETECTOR_BACKEND}. "
          f"Ctrl+C to quit. Logging to {log_path}\n")
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
                line = time.strftime('%H:%M:%S') + f"  {fps:5.1f} FPS   {labels}"
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
                n, t0 = 0, time.monotonic()
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()
        log.close()


if __name__ == '__main__':
    main()
