"""
Focus the lens by a number, not by eye.

    python3 -m tools.focus            centre region, 5 readings a second
    python3 -m tools.focus --hz 10    faster, for fine adjustment
    python3 -m tools.focus --full     score the whole frame, not the centre

Turn the barrel slowly and maximise SHARP. The peak-hold column remembers
the best score seen, so you can turn past the peak and come back to it.

Sharpness is the variance of the Laplacian: a real focus metric, high on
crisp edges and low on blur. The absolute number is meaningless, it depends
on the scene. Only the direction it moves matters.

TARGET: something with fine detail at the distance you care about, a page of
text or a keyboard, at one to two metres. NOT the traffic light prop: a lamp
is a blob at every focus setting and tells you nothing.

This tool deliberately runs the camera on AUTO exposure, overriding
CAMERA_EXPOSURE_US. The run exposure of 4000 us renders a room nearly black
on purpose, and you cannot focus on a black frame. Focus is a property of
the optics and carries over to any exposure.

Why focus at all when the robot moves: on this sensor, at this focal length
and this processing resolution, depth of field is enormous. Focused near one
metre, everything from roughly half a metre to infinity is sharp at the
resolution the detector works in. It is one setting for the whole course.
The failure it prevents is a module that shipped focused at 20 cm, which
looks perfect on the bench at arm's length and blurs every lamp on the track.

When done, LOCK IT: a dot of nail polish or glue on the thread. A barrel
that turns freely will drift on a robot that vibrates.
"""
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

import config
from hardware.camera import make_camera
from tools.sessionlog import SessionLog


def sharpness(frame_bgr, centre=True):
    """Variance of the Laplacian. Higher is sharper."""
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if centre:
        # The centre 60%: the corners of a cheap lens are soft however well
        # it is focused, and including them flattens the peak you are hunting.
        h, w = grey.shape
        grey = grey[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def bar(value, peak, width=40):
    if peak <= 0:
        return ''
    return '#' * max(0, min(width, int(width * value / peak)))


def main():
    args = sys.argv[1:]
    hz = float(args[args.index('--hz') + 1]) if '--hz' in args else 5.0
    centre = '--full' not in args
    # Focus needs a visible scene; the run exposure does not provide one.
    config.CAMERA_EXPOSURE_US = None
    config.CAMERA_LOCK_AE = False
    config.CAMERA_LOCK_AWB = False
    cam = make_camera(config)
    log = SessionLog('focus', f"region={'centre' if centre else 'full'} hz={hz}")
    print('Turn the barrel slowly and maximise SHARP. Ctrl+C when the peak stops moving.')
    print('Aim at fine detail (text, a keyboard) at one to two metres, NOT at the lamp.\n')
    peak = 0.0
    last_id = -1
    period = 1.0 / hz
    next_print = 0.0
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
            value = sharpness(frame, centre)
            peak = max(peak, value)
            print(log.line(f"SHARP {value:8.1f}   peak {peak:8.1f}  {bar(value, peak)}",
                           echo=False), flush=True)
    except KeyboardInterrupt:
        print(f"\nbest seen: {peak:.1f}. Lock the barrel with a dot of glue.")
    finally:
        cam.close()
        log.close()


if __name__ == '__main__':
    main()
