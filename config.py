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
# csi only: freeze auto white balance and auto exposure after a 1 s warmup.
# Both default ON. AWB live lets a big colored prop drag every hue with it.
# AE live is worse: re-aiming the camera re-meters the scene, so whether a
# lamp clears a detector brightness gate depends on framing rather than on
# the lamp. Bench 2026-09-01: a red lamp detected only after the camera was
# nudged upward, and every threshold comparison that session was taken
# against a moving exposure.
CAMERA_LOCK_AWB = True
CAMERA_LOCK_AE = True

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
STOPSIGN_WHITE_FRAC = 0.06     # white STOP text/border separates sign from lamp
