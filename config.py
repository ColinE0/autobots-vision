"""
Tunables for the sensing code, off the robot.

Same names and same values as the robot's config.py, so a module written here
drops into the flight code without an edit. Nothing imports this at run time
on the bot: the robot passes its own config in. Add a new name here and it has
to be added on the robot side too, or the module raises AttributeError on the
Pi.
"""

# Forward rangefinder: obstacle (other-bot) avoidance, hardware/ranger.py.
# OFF BY DEFAULT until the sensor is on the bench. Part as bought 2026-08-28:
# Starry GY-VL53L0XV2 breakout, VL53L0X ToF, 940 nm, 2.8-5 V, I2C, 5 per pack
# (~2 m; on a dark 17% grey target more like 0.8 to 1.2 m, still far past the
# 0.60 m slow line). Sits on the I2C bus the INA219 already uses (INA219 0x40,
# VL53L0X 0x29), so no GPIO. Every unit in the pack boots at 0x29, so exactly
# one goes on the bus: a second would collide and needs holding in XSHUT reset
# at boot to be re-addressed. The other four are spares, not a sensor array.
# Mount it forward-facing at bumper height, LEVEL, never tilted down (the L0X
# has no ROI trim, so floor-in-the-cone is a mechanical fix: tilt up a degree
# or raise the mount).
RANGE_ENABLED = False
RANGE_I2C_BUS = 1
RANGE_I2C_ADDR = 0x29
# VL53L0X accuracy mode, FALLBACK ONLY: ranger.py resolves LONG_RANGE by
# name from the installed binding. The binding's constants are 0-BASED
# (0 GOOD 33 ms / 1 BETTER 66 / 2 BEST 200 / 3 LONG_RANGE 33 /
# 4 HIGH_SPEED 20); the 1-based table recorded here before was wrong,
# caught at first bench contact 2026-09-04 (RANGE_MODE = 4 would have
# meant HIGH_SPEED). Used only if the binding exposes no LONG_RANGE name.
RANGE_MODE = 3
RANGE_POLL_HZ = 20
RANGE_EMA_ALPHA = 0.5         # light smoothing; the sensor is already mm-quiet
RANGE_MAX_M = 2.0             # reads beyond this count as "nothing ahead"
RANGE_SLOW_M = 0.60           # start slowing here
RANGE_HOLD_M = 0.25           # hold here
RANGE_HYST_M = 0.10           # must clear HOLD + this to resume
RANGE_SLOW_SCALE = 0.40       # forward speed multiplier at the hold line
RANGE_CLEAR_S = 0.75          # clear this long before resuming
RANGE_STALE_S = 1.0           # no valid read this long = ignore the ranger

# Camera detection: vision/. Frames are downscaled to this width before any
# detection work (full-resolution frames don't fit the Pi Zero 2W's budget).
# Same name and value as the robot's config. Bring-up check: the stop sign's
# octagon corners must still resolve at this width; raise it only with FPS
# evidence from the actual Pi.
DETECT_PROC_WIDTH = 320

# Camera for tools/detect_preview.py, same names and values as the robot:
#   'csi'  Pi camera (Arducam IMX219) via picamera2 (apt: python3-picamera2)
#   'usb'  laptop / USB webcam via OpenCV; CAMERA_INDEX applies only here
#          (index 2 on the laptop the detectors were tuned on)
# 320x240 capture on purpose: it is the detector's working width, so the
# CSI ISP scales for free and the downscale becomes a no-op.
CAMERA_BACKEND = 'csi'
CAMERA_INDEX = 0             # usb only
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30
CAMERA_USE_MJPG = True       # usb only: MJPG keeps USB bandwidth sane
# csi only: the SENSOR mode, distinct from the main stream above. Unset,
# picamera2 picks the smallest mode that covers CAMERA_WIDTH x CAMERA_HEIGHT,
# which on the IMX219 is 640x480, a (1000,752)/1280x960 CENTRE CROP: about
# 2.5x tighter than the lens, so a level camera at bumper height cannot hold
# a 15 cm sign or the 14 cm lane in frame one block out. Confirmed on the
# flight Zero 2026-09-08 (rpicam-hello --list-cameras; picamera2 raw=640x480).
# 1640x1232 is the 2x2-binned FULL frame ((0,0)/3280x2464, 81 fps ceiling);
# the ISP still scales it to the main size for free. Area-fraction thresholds
# (DETECT_MIN_AREA_FRAC and friends) were tuned on the crop: a prop reads
# about 6.6x smaller here. None = the old picamera2 choice.
CAMERA_SENSOR_SIZE = (1640, 1232)
# csi only: freeze auto white balance and auto exposure after a 1 s warmup.
# Both default ON. AWB live lets a big colored prop drag every hue with it.
# AE live is worse: re-aiming the camera re-meters the scene, so whether a
# lamp clears a detector brightness gate depends on framing rather than on
# the lamp. Bench 2026-09-01: a red lamp detected only after the camera was
# nudged upward, and every threshold comparison that session was taken
# against a moving exposure.
CAMERA_LOCK_AWB = True
CAMERA_LOCK_AE = True
# csi only: how auto-exposure METERS before the lock, and again at the START
# relock. 'highlight' picks the imx219.json rpi.agc constraint set that adds
# an upper bound holding the brightest 2% of pixels at 0.8 of full scale, so
# a lit lamp in frame pulls the exposure down instead of clipping to white.
# A clipped lamp loses its colour: the detector's saturation floor deletes
# the white core and what is left read as STOP text (review 2026-09-05; the
# clipped-core test below is the detector-side guard, this is the cause).
# 'normal' is libcamera's default metering; 'shadows' biases the other way.
CAMERA_AE_CONSTRAINT = 'highlight'
# Exposure offset in stops (log2) on the metered target, honoured only while
# AE runs. 0.0 trusts the constraint. Floored at -1.0 in hardware/camera.py:
# a V 180 printed sign one stop under reads about V 90, still over SIGN_V_MIN.
CAMERA_EV = 0.0

# Run recording (csi only). Records what the DETECTOR sees, through the Pi's
# hardware H.264 encoder on a second stream, so the ARM cores never touch the
# pixels and the frame rate barely moves. The recording stream is deliberately
# the same size as the main stream: a run is worth keeping because it shows
# what the detector was handed, not because it is pretty. Off by default;
# tools/test_camera.py --record turns it on for one run.
# The camera is mounted upside down on the chassis, confirmed 2026-09-11 from a
# saved frame: the scene arrives rotated 180. Corrected on the ISP, which is
# free, rather than with a cv2.flip on every frame, which is not. It must be
# BOTH axes. A vertical flip alone mirrors left and right, and
# DETECT_IGNORE_BOTTOM_FRAC and the detector's centring gates would then be
# quietly wrong instead of loudly wrong. Set False if the camera is remounted.
CAMERA_ROTATE_180 = True

CAMERA_RECORD = False
CAMERA_RECORD_BITRATE = 1_500_000   # ~0.2 MB/s at 320x240, ~11 MB a minute
CAMERA_RECORD_DIR = 'recordings'    # gitignored; files are raw .h264

# Detector backend, picked by vision.detector.make_detector():
#   'classical'  HSV masks + contour gates + white-content stop/lamp split
#                (the robot's detector; 10-20 FPS on the Zero 2 W)
#   'geometric'  the octagon-fit + glow-profile detectors in this repo
# Both speak detect(frame) -> [Detection(label, area_frac)]; A/B them on the
# real camera with tools/test_camera.py and flip this one line.
DETECTOR_BACKEND = 'classical'
DETECT_IGNORE_BOTTOM_FRAC = 0.25   # bottom of frame is floor/line, not signs
LIGHT_S_MIN = 100   # HSV saturation floor for a lamp; rejects washed-out grey
LIGHT_V_MIN = 170   # HSV brightness floor for yellow/green masks, and the floor a
                    # blob's pixels must clear before the glow test judges them. Was
                    # 80, which a red/yellow/green lens clears in ordinary room light,
                    # so a 3-lens traffic-light module reported all three colours at
                    # once (bench, 2026-09-01). 170 matches the geometric MIN_VALUE.
SIGN_V_MIN = 80     # brightness floor for the RED mask. A printed stop sign is not
                    # a light source and must not have to clear a lamp's floor; it
                    # is told from a lamp by LAMP_GLOW, not by brightness alone.
# A lit lamp GLOWS: its coloured pixels sit near clipping. Per colour:
# (mean V, 90th-percentile V, share of pixels at/above the bright level, that
# bright level). The values are the geometric backend's, which held on all
# three lamps of the module at 13 in with exposure locked (bench 2026-09-02).
# Red is the strictest because a lit red lamp is the object most likely to be
# mistaken for a stop sign, and that mistake ends the run.
LAMP_GLOW = {
    'red':    (220, 240, 0.40, 220),
    'yellow': (200, 230, 0.30, 210),
    'green':  (185, 220, 0.25, 205),
}
LAMP_WINNER_RATIO = 1.2       # one lamp per frame: biggest must beat the next by this
DETECT_MIN_AREA_FRAC = 0.002  # ignore blobs smaller than 0.2% of the frame
CONFIRM_FRAMES_N = 3          # TemporalFilter window ...
CONFIRM_FRAMES_K = 2          # ... act on K of the last N frames
STOPSIGN_WHITE_FRAC = 0.06     # white STOP text/border separates sign from lamp:
                               # this share of the red blob's INTERIOR must be white
# What counts as white inside a red blob. A printed sign is a reflector, so a
# darker exposure dims its letters and its red together; the floor therefore
# follows the blob: V >= STOPSIGN_WHITE_REL x the median V of its red pixels,
# never under STOPSIGN_WHITE_V_MIN (a floor for near-black frames) and never
# over STOPSIGN_WHITE_V_CAP (clipped letters still count). A fixed V 170
# floor lost a rendered sign at 0.7x the tuning exposure and this rule held
# to 0.5x (review 2026-09-07); 0.7x is where the highlight AE constraint puts
# a sign once a lit lamp is in frame. Real sign white-to-red V ratio assumed
# 1.15 to 1.2; read it off a saved test_camera frame before trusting 1.1.
STOPSIGN_WHITE_REL = 1.1
STOPSIGN_WHITE_V_MIN = 100
STOPSIGN_WHITE_V_CAP = 245
# Flatness veto for the glow test, OFF (0) until read on a real lamp. A lit
# lamp falls off from centre to rim, a printed sign is flat, so a blob whose
# V spread (p90 minus p10 of the pixels the glow test judges) is under this
# is not a lamp however bright: flat red at V 240 clears every glow floor and
# reads as red_light, a hold that never clears. Rendered frames 2026-09-07:
# lamps 61 to 75, flat signs 0 to 4, so 30 is the value to try. Risk on the
# other side: the highlight constraint leaves a real lamp less clipped and
# flatter, and a misfire misses a red light. Read a real lamp first.
LAMP_MIN_V_SPREAD = 0
# A blown-out lamp (close, or metered under a long locked exposure) clips to
# white almost to its rim. The colour mask deletes that core (S near 0), and
# the ring plus halo that survive fail LAMP_GLOW on mean V, so the lamp used
# to fall through to the white test and read as a stop sign. The core the
# mask threw away is judged by SHAPE instead: the largest white component in
# the blob's bounding box must cover this share of the box, be round
# (width / height inside the range), filled (area over its own box) and
# centred (centroid within this fraction of the box from the box centre).
# STOP text is a 3.5:1 band and a sign's white border is hollow, so a printed
# sign never passes. Review 2026-09-05: held on 15 synthetic cases, no sign
# regression; real-frame check owed at the matched re-run.
LAMP_CORE_MIN_FRAC = 0.10
LAMP_CORE_ASPECT = (0.6, 1.6)
LAMP_CORE_MIN_EXTENT = 0.55
LAMP_CORE_CENTER_TOL = 0.25
# The core must be genuinely clipped: V at or above this, an absolute floor
# independent of the sign's relative letter floor above. A grey V 150
# reflection on an unlit lens cleared the relative floor, passed the shape
# test and was a red_light (review 2026-09-08). A lit LED core is at the top
# of the scale whatever the exposure.
LAMP_CORE_V_MIN = 235
# The glow test's bright share is over ALL of a blob's masked pixels and
# needs at least this many bright pixels: a dark lens with one 2 px specular
# highlight (13 clipped pixels) read as red_light when only the pixels above
# LIGHT_V_MIN were judged (review 2026-09-08).
LAMP_MIN_BRIGHT_PIXELS = 16
# STOP text is a row of at least three letter-sized white marks; each mark
# must be strokes, not a filled dot: its area over its box under this. A
# clipped LED emitter is a solid disc (extent 0.79); Hershey and Highway
# Gothic letters run 0.3 to 0.6. Without the cap a lamp with three emitters
# in a row read as a stop sign, the run-ending failure of 2026-09-02 again
# (review 2026-09-08, synthetic). Raise toward 0.8 only if a real sign at
# the far end of its working distance is lost because its letters blur
# into solid blobs; check that with tools/test_camera.py --save first.
STOPSIGN_LETTER_MAX_EXTENT = 0.70
# Geometric backend (vision/detector_geometric.py): the tunables it shares
# with the classical path it takes by name (LIGHT_V_MIN, DETECT_MIN_AREA_FRAC,
# LAMP_GLOW). This one is its own: the largest blob its traffic-light path
# considers. It shipped at 0.05, which dropped every close lamp (PR #1 item
# 1, 2026-09-05); 0.35 keeps a lamp filling a third of the frame.
GEO_LIGHT_MAX_AREA_FRAC = 0.35
