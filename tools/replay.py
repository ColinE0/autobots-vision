"""
Replay a recorded run through the detector, off the robot.

    python -m tools.replay run.mp4
    python -m tools.replay run.mp4 --raw
    python -m tools.replay run.mp4 --frames 1180 1240

Takes any clip the Pi can produce, e.g. rpicam-vid -t 30000 -o run.h264 then
ffmpeg -r 30 -i run.h264 -c copy run.mp4. Answers what a log line cannot: whether
a 0.5% blob was the prop, a reflection, or a lamp the gates nearly rejected,
and whether a confirmed label FLAPPED rather than holding.

The flap count is the number the console cannot give you. A label that turns
on and off six times in a second is not a detection, it is a threshold sitting
on the edge of the signal, and it reads identically to a real detection in a
scrolling log.

--raw reports per-frame detections instead of what TemporalFilter confirms.
--frames A B writes those frames to frames/ as PNG so a single moment can be
opened, measured, and argued about with a picture rather than a number.
"""
import statistics
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

import config
from vision.detector import make_detector, TemporalFilter

LABELS = ('stop_sign', 'red_light', 'yellow_light', 'green_light')


def iter_video(path):
    """Yield BGR frames from a video file. Raw .h264 works if ffmpeg does."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}. For raw .h264, wrap it first: "
                         f"ffmpeg -r 30 -i in.h264 -c copy out.mp4")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def analyse(frames, cfg, raw=False, det=None, on_frame=None):
    """Per-label accounting over a whole recording.

    Returns {label: {...}} plus a 'frames' total. `det` is injectable so the
    accounting can be tested without running the real detector, and
    `on_frame(i, frame, labels)` is the hook the CLI uses to dump stills.
    """
    det = det if det is not None else make_detector(cfg)
    filt = None if raw else TemporalFilter(cfg)
    stats = {lbl: {'frames': 0, 'transitions': 0, 'longest': 0, 'areas': []}
             for lbl in LABELS}
    was_on = {lbl: False for lbl in LABELS}
    streak = {lbl: 0 for lbl in LABELS}
    total = 0
    for i, frame in enumerate(frames):
        total += 1
        dets = det.detect(frame)
        if filt is None:
            shown = {d.label: d.area_frac for d in dets}
        else:
            filt.update(dets)
            shown = {lbl: filt.area(lbl) for lbl in LABELS if filt.confirmed(lbl)}
        for lbl in LABELS:
            on = lbl in shown
            s = stats[lbl]
            if on:
                s['frames'] += 1
                area = shown[lbl]
                if area is not None:
                    s['areas'].append(area)
                streak[lbl] += 1
                s['longest'] = max(s['longest'], streak[lbl])
            else:
                streak[lbl] = 0
            if on != was_on[lbl]:
                s['transitions'] += 1
                was_on[lbl] = on
        if on_frame is not None:
            on_frame(i, frame, sorted(shown))
    stats['frames'] = total
    return stats


def report(stats, fps):
    total = stats['frames']
    print(f"\n{total} frames, {total / fps:.1f} s at {fps:g} fps\n")
    print(f"{'label':<13}{'frames':>8}{'share':>8}{'flaps':>7}"
          f"{'longest':>9}{'area min/med/max':>26}")
    for lbl in LABELS:
        s = stats[lbl]
        if not s['frames']:
            continue
        a = s['areas']
        area = (f"{min(a):.2%} / {statistics.median(a):.2%} / {max(a):.2%}"
                if a else '-')
        # transitions counts both edges; a flap is an on/off pair.
        print(f"{lbl:<13}{s['frames']:>8}{s['frames'] / total:>7.1%}"
              f"{s['transitions'] // 2:>7}{s['longest'] / fps:>8.2f}s{area:>26}")
    if not any(stats[l]['frames'] for l in LABELS):
        print("  nothing detected in the whole recording")


def main():
    args = [a for a in sys.argv[1:]]
    raw = '--raw' in args
    dump = None
    if '--frames' in args:
        i = args.index('--frames')
        dump = (int(args[i + 1]), int(args[i + 2]))
        del args[i:i + 3]
    paths = [a for a in args if not a.startswith('--')]
    if len(paths) != 1:
        raise SystemExit(__doc__)
    path = pathlib.Path(paths[0])
    fps = float(getattr(config, 'CAMERA_FPS', 30))
    frames_dir = ROOT / 'frames'
    hook = None
    if dump:
        frames_dir.mkdir(exist_ok=True)

        def write_still(i, frame, labels):
            if dump[0] <= i <= dump[1]:
                name = f"replay_{i:06d}_{'+'.join(labels) or 'none'}.png"
                cv2.imwrite(str(frames_dir / name), frame)

        hook = write_still

    view = 'raw' if raw else f"confirmed({config.CONFIRM_FRAMES_K}of{config.CONFIRM_FRAMES_N})"
    print(f"replaying {path.name}  detector={config.DETECTOR_BACKEND}  view={view}")
    stats = analyse(iter_video(path), config, raw=raw, on_frame=hook)
    report(stats, fps)
    if dump:
        print(f"frames {dump[0]}-{dump[1]} written to {frames_dir}")


if __name__ == '__main__':
    main()
