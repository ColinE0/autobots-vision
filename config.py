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
# VL53L0X accuracy mode (timing budget must fit the 50 ms poll period):
# 1 GOOD 33 ms / 2 BETTER 66 ms / 3 BEST 200 ms / 4 LONG_RANGE 33 ms /
# 5 HIGH_SPEED 20 ms. LONG_RANGE: full ~2 m reach at a budget that fits.
# Confirm the constants against the installed binding at bring-up.
RANGE_MODE = 4
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
