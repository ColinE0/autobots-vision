"""
Camera + detector smoke test. Prints detections and FPS. No display needed
(headless). Point the camera at a stop sign / colored light to verify.
On the CSI camera: no red hits on a real stop sign usually means bad color
order (see the RGB888 note in hardware/camera.py), and a soft image means
the lens barrel needs a twist (the Arducam IMX219 focuses by rotating it).

    python3 -m tools.test_camera          confirmed view: what the pilot acts on
    python3 -m tools.test_camera --raw    unfiltered per-frame detector output
    python3 -m tools.test_camera --no-save  do not keep stills this run
    python3 -m tools.test_camera --exposure 4000   pin exposure, in us

The default runs detections through the robot's TemporalFilter (K-of-N
confirmation) so what prints is what the pilot would actually see. --raw
bypasses it and shows every per-frame result, which is far noisier and is a
debugging view, not a measure of detector quality: single-frame flips are
what the filter exists to absorb.

Detections print when the SET OF LABELS changes, not on a timer, so a
hand-held prop reads as transitions rather than a 1-in-60 sample. FPS is
reported separately every 2 s. Lines are timestamped and appended to
test_camera.log, whose session header carries the git revision, both
backends, resolution, view mode and the locked exposure.

Stills are kept AUTOMATICALLY: the frame behind each label change is written
to frames/ (gitignored) as a PNG named by time, frame id and labels, so a
surprising detection can be re-run through the detector on a laptop instead of
argued about from a log line. On change only, never every frame. Works in both
views. --no-save turns it off, and CAMERA_SAVE_MAX_FRAMES caps one run so an
unattended session cannot fill the card.
"""
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import config
from hardware.camera import make_camera
from vision.detector import make_detector, TemporalFilter
from tools.sessionlog import SessionLog

# The complete label vocabulary both backends emit. TemporalFilter is asked
# about labels by name, so the tool needs the list rather than the filter's
# internal window.
LABELS = ('stop_sign', 'red_light', 'yellow_light', 'green_light')


def main():
    args = sys.argv[1:]
    raw = '--raw' in args
    save = getattr(config, 'CAMERA_SAVE_FRAMES', True) and '--no-save' not in args
    if '--exposure' in args:
        # Venue calibration knob. Sweep it here rather than editing config.py
        # between runs, then write the value that works into the config.
        config.CAMERA_EXPOSURE_US = int(args[args.index('--exposure') + 1])
    frames_dir = ROOT / 'frames'
    if save:
        frames_dir.mkdir(exist_ok=True)
    cam = make_camera(config)
    det = make_detector(config)
    filt = None if raw else TemporalFilter(config)
    view = ('raw' if raw else
            f"confirmed({config.CONFIRM_FRAMES_K}of{config.CONFIRM_FRAMES_N})")
    # Frozen exposure / gain when the camera locked them. Two runs are only
    # comparable if they were shot at the same exposure, so it goes in the
    # header alongside the git rev.
    locked = getattr(cam, 'locked', {})
    lock_note = ' '.join(f"{k}={v}" for k, v in sorted(locked.items())) or 'auto'
    # How AE metered before it was locked (csi only); the lit-lamp fix of
    # 2026-09-05 lives here, so a run must say which mode it was shot under.
    meter_note = (f"{config.CAMERA_AE_CONSTRAINT} ev={config.CAMERA_EV}"
                  if config.CAMERA_BACKEND == 'csi' else 'n/a')
    # Sensor mode (csi only). Runs before 2026-09-08 logged only the main
    # size and were shot on the 640x480 centre crop; a run on the full frame
    # is not comparable to them, so the header says which.
    sensor = getattr(cam, 'sensor_size', None)
    sensor_note = (('auto' if sensor is None else f"{sensor[0]}x{sensor[1]}")
                   if config.CAMERA_BACKEND == 'csi' else 'n/a')
    log = SessionLog('test_camera',
                     f"camera={config.CAMERA_BACKEND} "
                     f"detector={config.DETECTOR_BACKEND} "
                     f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}@{config.CAMERA_FPS} "
                     f"sensor={sensor_note} "
                     f"view={view} meter=[{meter_note}] exposure=[{lock_note}] "
                     f"save={save}")
    print(f"camera={config.CAMERA_BACKEND}  detector={config.DETECTOR_BACKEND}  "
          f"view={view}. Ctrl+C to quit. Logging to {log.path}"
          + (f", frames to {frames_dir}" if save else '')
          + "\n")
    last_id, n, t0 = -1, 0, time.monotonic()
    saved, cap = 0, int(getattr(config, 'CAMERA_SAVE_MAX_FRAMES', 200))
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
            if filt is None:
                shown = [(d.label, d.area_frac) for d in dets]
            else:
                filt.update(dets)
                shown = [(lbl, filt.area(lbl)) for lbl in LABELS
                         if filt.confirmed(lbl)]
            # Key on the labels alone. area_frac jitters every frame, so
            # keying on the printed string would emit at the frame rate.
            key = tuple(lbl for lbl, _ in shown)
            if key != last_key:
                labels = ', '.join(
                    f"{lbl}({frac*100:.1f}%)" for lbl, frac in shown) or '-'
                print(log.line(labels, echo=False), flush=True)
                if save and saved < cap:
                    name = (f"{time.strftime('%Y%m%d-%H%M%S')}_{fid}_"
                            f"{'+'.join(key) or 'none'}.png")
                    cv2.imwrite(str(frames_dir / name), frame)
                    saved += 1
                    if saved == cap:
                        # Said once, not per frame: a run flapping at a
                        # threshold would otherwise bury its own detections.
                        print(log.line(f'still cap {cap} reached, no more saved'),
                              flush=True)
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
