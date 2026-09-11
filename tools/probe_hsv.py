"""
Live HSV probe: prints what the classical detector measures, not what it
decides. For finding the LIGHT_S_MIN / LIGHT_V_MIN / SIGN_V_MIN / LAMP_GLOW
sweet spot against the real props in the real room.

    python3 -m tools.probe_hsv            2 lines per second
    python3 -m tools.probe_hsv --hz 5     faster
    python3 -m tools.probe_hsv --save     also keep a frame every 2 s in frames/
    python3 -m tools.probe_hsv --sweep    walk fixed exposures, print a table
    python3 -m tools.probe_hsv --sweep 16000,8000,4000,2000 --dwell 8

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

--sweep walks a list of FIXED exposures, holding each for --dwell seconds
(default 8), and prints one row per exposure per colour: how often a blob
existed, its median area, saturation and brightness, its bright and core pixel
counts, the gate it failed most often, and how often the real detector confirmed
the label. It reopens the camera at each step, so every row is exactly what a run
at that exposure would see.

What the sweep is looking for is a WINDOW with two walls. Too long and ambient
returns: the lamp merges with its own reflection and the merged blob fails the
glow test, taking the lamp with it. Too short and the lamp clips to white, loses
saturation, and the colour mask deletes it. Read the S column for the lower wall
and the area column for the upper one: area jumping several-fold between two steps
is the reflection rejoining the lamp. If no exposure satisfies both, one fixed
value cannot serve the lamp and alternating exposures is the answer. Run it twice,
once at normal room light and once with every light on, WITHOUT moving the prop or
the camera: a winning value that moves between the two is not venue-independent.

Reading it: a lit lamp should print GLOWS in its colour. An unlit lens or a
printed sign should print a fail, and the fail should be a comfortable
margin, not a hair. If a lit lamp fails on "bright share" or "mean V", the
exposure is probably too high for LIGHT_V_MIN (the lamp is not clipping) or
the lens is dim; before touching LAMP_GLOW check the exposure line in the
header and compare with a test_camera run.

Same exposure lock as test_camera: have the prop lit and in frame before
launch. Logged to probe_hsv.log with the same header fields.
"""
import statistics
from collections import Counter
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / 'frames'
sys.path.insert(0, str(ROOT))
import cv2
import config
from hardware.camera import make_camera
from vision.detector import make_detector
from vision.probe import Probe, centre_hsv, format_line
from tools.sessionlog import SessionLog

COLOURS = ('red', 'yellow', 'green')
# The probe speaks in colours, the detector in labels. Red is deliberately
# mapped from BOTH red_light and stop_sign: the red mask is shared, and a sweep
# that counted only one would look like the lamp vanished when the split moved.
LABEL_TO_COLOUR = {'red_light': 'red', 'stop_sign': 'red',
                   'yellow_light': 'yellow', 'green_light': 'green'}


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def sweep(exposures, dwell, log):
    """Hold each fixed exposure for dwell seconds and tabulate what it gives."""
    rows = []
    scenes = []
    for us in exposures:
        config.CAMERA_EXPOSURE_US = us
        # Reopened per step rather than set live, so each row is exactly what a
        # run at that exposure would see, first frame included.
        cam = make_camera(config)
        probe = Probe(config)
        det = make_detector(config)
        seen = {c: [] for c in COLOURS}
        centre = []
        peak = []
        clip_px = []
        clip_s = []
        fails = {c: [] for c in COLOURS}
        confirmed = {c: 0 for c in COLOURS}
        frames = 0
        last_id = -1
        last_frame = None
        deadline = time.monotonic() + dwell
        try:
            while time.monotonic() < deadline:
                fid, frame = cam.read()
                if fid == last_id:
                    time.sleep(0.005)
                    continue
                last_id = fid
                frames += 1
                centre.append(centre_hsv(frame))
                # Brightest pixel anywhere. With no blob this is the only
                # evidence of whether the frame held anything at all.
                peak.append(int(frame.max()))
                # A clipped EMITTER goes white: V at 255 with the colour
                # washed out of it. Counting those pixels and reading their
                # saturation separates 'lamp too bright for this exposure'
                # from 'nothing lit', which peak brightness alone cannot.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = hsv[:, :, 2] >= 250
                n = int(mask.sum())
                clip_px.append(n)
                # Mean saturation over the clipped pixels is useless: the
                # coloured ring around a white core is clipped too and drags
                # the average up. What separates the cases is the SHARE of
                # clipped pixels that have lost their colour entirely.
                sat = hsv[:, :, 1][mask]
                clip_s.append(float((sat < 60).mean()) if n else None)
                stats = probe.measure(frame)
                for colour in COLOURS:
                    st = stats.get(colour)
                    if st:
                        seen[colour].append(st)
                        if st.get("fail"):
                            fails[colour].append(st["fail"])
                for d in det.detect(frame):
                    key = LABEL_TO_COLOUR.get(d.label)
                    if key:
                        confirmed[key] += 1
                last_frame = frame
        finally:
            cam.close()
        if last_frame is not None:
            FRAMES_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(FRAMES_DIR / f'sweep_{us}us.png'), last_frame)
        scene = {
            'us': us, 'frames': frames,
            'centre_v': _median([c[2] for c in centre]),
            'peak_v': _median(peak),
            'clip_px': _median(clip_px),
            'clip_white': _median(clip_s),
        }
        scenes.append(scene)
        for colour in COLOURS:
            st = seen[colour]
            if not st and not confirmed[colour]:
                continue
            worst = Counter(fails[colour]).most_common(1)
            rows.append({
                "us": us, "colour": colour, "frames": frames,
                "blob": len(st), "detect": confirmed[colour],
                "area": _median([x.get("area_frac") for x in st]),
                "s": _median([x.get("s_mean") for x in st]),
                "v": _median([x.get("v_mean") for x in st]),
                "bright": _median([x.get("bright") for x in st]),
                "core": _median([x.get("core_px") for x in st]),
                "fail": worst[0][0] if worst else "-",
            })
        log.line(f"exposure {us} us: {frames} frames")
    return rows, scenes


def _fmt(v):
    return '-' if v is None else f'{v:.0f}'


def report_sweep(rows, scenes=(), log=None):
    header = (f"{'exposure':>9}{'col':>4}{'blob%':>7}{'det%':>6}"
              f"{'area':>8}{'S':>6}{'V':>6}{'bright':>8}{'core':>6}"
              "  most common failure")
    print()
    print(log.line(header, echo=False), flush=True)
    for r in rows:
        frames = r['frames'] or 1
        area = '-' if r['area'] is None else f"{r['area']:.2%}"
        line = (f"{r['us']:>9}{r['colour'][0].upper():>4}"
                f"{r['blob'] / frames:>7.0%}{r['detect'] / frames:>6.0%}"
                f"{area:>8}{_fmt(r['s']):>6}{_fmt(r['v']):>6}"
                f"{_fmt(r['bright']):>8}{_fmt(r['core']):>6}  {r['fail']}")
        print(log.line(line, echo=False), flush=True)
    if not rows:
        # An empty table is a result too, but it has to say which result.
        # Without the scene numbers there is no way to tell "no prop in
        # frame" from "prop there, every exposure too short".
        print(log.line('  no red, yellow or green blob at ANY exposure.', echo=False),
              flush=True)
    if scenes:
        print()
        print(log.line(f"{'exposure':>9}{'frames':>8}{'centre V':>10}{'peak V':>8}"
                       f"{'clipped':>9}{'white%':>8}   scene", echo=False), flush=True)
        for s in scenes:
            peak = s['peak_v'] or 0
            clip = s.get('clip_px') or 0
            white = s.get('clip_white')
            if peak < 40:
                note = 'black: nothing lit, or every exposure far too short'
            elif clip and white is not None and white > 0.3:
                note = 'clipped to WHITE: an emitter too bright for this exposure'
            elif clip:
                note = 'clipping but still coloured: a lamp the mask can see'
            else:
                note = 'dim: no clipped highlight, so no lamp core to find'
            print(log.line(f"{s['us']:>9}{s['frames']:>8}{_fmt(s['centre_v']):>10}"
                           f"{_fmt(s['peak_v']):>8}{_fmt(clip):>9}"
                           f"{'-' if white is None else format(white, '.0%'):>8}"
                           f"   {note}", echo=False), flush=True)
    print()


def main():
    args = sys.argv[1:]
    save = '--save' in args
    hz = 2.0
    if '--hz' in args:
        hz = float(args[args.index('--hz') + 1])
    if '--exposure' in args:
        config.CAMERA_EXPOSURE_US = int(args[args.index('--exposure') + 1])
    if '--sweep' in args:
        i = args.index('--sweep')
        nxt = args[i + 1] if len(args) > i + 1 else ''
        steps = ([int(x) for x in nxt.split(',')] if nxt and not nxt.startswith('--')
                 else [16000, 8000, 4000, 2000])
        dwell = float(args[args.index('--dwell') + 1]) if '--dwell' in args else 8.0
        log = SessionLog('probe_hsv', f'sweep={steps} dwell={dwell}')
        print(f'sweeping {steps} us, {dwell:g}s each. Hold the prop still.')
        try:
            report_sweep(*sweep(steps, dwell, log), log=log)
        finally:
            log.close()
        return
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
