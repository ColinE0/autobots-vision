"""
Live HSV probe: prints what the classical detector measures, not what it
decides. For finding the LIGHT_S_MIN / LIGHT_V_MIN / SIGN_V_MIN / LAMP_GLOW
sweet spot against the real props in the real room.

    python3 -m tools.probe_hsv            2 lines per second
    python3 -m tools.probe_hsv --hz 5     faster
    python3 -m tools.probe_hsv --save     also keep a frame every 2 s in frames/

Each line:

    centre H  0 S 61 V 88 | R: 1.3% S142 V95 p90 118 br 0.00/0 core 0 [bright pixels 0 < 16] | Y: - | G: -

  centre   median H, S, V of a 20 px patch at the frame centre. Hold the prop
           in the crosshair and read its colour directly (OpenCV scale: H 0
           to 180, S and V 0 to 255).
  R/Y/G    the largest blob of that colour the detector would consider:
           area as a % of the frame, mean S, mean V, 90th-percentile V, the
           bright share and count (pixels at/above the LAMP_GLOW bright
           level), the count of clipped white pixels inside the blob
           (S under WHITE_S_MAX, V at/above LAMP_CORE_V_MIN), then the
           verdict in brackets: GLOWS; CORE (...) when the glow test failed
           for the reason given but the blown-out white core passes the
           shape test, which is still a lamp to the detector; or the FIRST
           glow gate the blob fails, in config units, so the number to move
           is named. "-" means no blob cleared the colour mask and the area
           floor; "shape ..." means the biggest one failed the shape gate.

Reading it: a lit lamp should print GLOWS in its colour. An unlit lens or a
printed sign should print a fail, and the fail should be a comfortable
margin, not a hair. If a lit lamp fails on "bright share" or "mean V", the
exposure is probably too high for LIGHT_V_MIN (the lamp is not clipping) or
the lens is dim; before touching LAMP_GLOW check the exposure line in the
header and compare with a test_camera run.

Same exposure lock as test_camera: have the prop lit and in frame before
launch. Logged to probe_hsv.log with the same header fields.
"""
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import config
from hardware.camera import make_camera
from vision.probe import Probe, centre_hsv, format_line
from tools.sessionlog import SessionLog


def main():
    args = sys.argv[1:]
    save = '--save' in args
    hz = 2.0
    if '--hz' in args:
        hz = float(args[args.index('--hz') + 1])
    frames_dir = ROOT / 'frames'
    if save:
        frames_dir.mkdir(exist_ok=True)
    cam = make_camera(config)
    probe = Probe(config)
    locked = getattr(cam, 'locked', {})
    lock_note = ' '.join(f"{k}={v}" for k, v in sorted(locked.items())) or 'auto'
    log = SessionLog('probe_hsv',
                     f"camera={config.CAMERA_BACKEND} "
                     f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}@{config.CAMERA_FPS} "
                     f"exposure=[{lock_note}] "
                     f"S_MIN={config.LIGHT_S_MIN} V_MIN={config.LIGHT_V_MIN} "
                     f"SIGN_V_MIN={config.SIGN_V_MIN} "
                     f"glow={config.LAMP_GLOW} hz={hz} save={save}")
    print(f"camera={config.CAMERA_BACKEND}  exposure=[{lock_note}]  "
          f"S_MIN={config.LIGHT_S_MIN} V_MIN={config.LIGHT_V_MIN} "
          f"SIGN_V_MIN={config.SIGN_V_MIN}. Ctrl+C to quit. Logging to {log.path}\n")
    period = 1.0 / hz
    last_id = -1
    next_print = 0.0
    next_save = 0.0
    try:
        while True:
            fid, frame = cam.read()
            if fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            now = time.monotonic()
            if now < next_print:
                continue
            next_print = now + period
            line = format_line(centre_hsv(frame), probe.measure(frame))
            print(log.line(line, echo=False), flush=True)
            if save and now >= next_save:
                next_save = now + 2.0
                cv2.imwrite(str(frames_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{fid}_probe.png"),
                            frame)
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()
        log.close()


if __name__ == '__main__':
    main()
