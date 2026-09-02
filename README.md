# autobots-vision

Sensor code for our senior design robot (AutoBots, Team 2.09). This is where we
put the camera work and the forward rangefinder so we can both pull it, run it,
and not overwrite each other. The rest of the robot code lives separately.

## Layout

```
hardware/   sensor drivers: the VL53L0X time-of-flight rangefinder, and
            camera.py (CSI picamera2 + USB webcam backends behind one factory)
vision/     the camera modules. TWO detector backends behind
            vision/detector.py's make_detector(), picked by
            config.DETECTOR_BACKEND: 'classical' (HSV + contour gates +
            white-content stop/lamp split, the robot's default) and
            'geometric' (octagon fit + glow-profile validation, in
            stop_sign_detector.py / traffic_light_detector.py behind
            detector_geometric.py). Both speak
            detect() -> [Detection(label, area_frac)]
tests/      whatever we can run without the robot
tools/      bench scripts. tools/range_check.py is the live ToF readout,
            tools/detect_preview.py is the live detection preview (geometric,
            draws boxes), tools/test_camera.py is the backend-switched
            camera + detector smoke test (prints FPS and labels)
```

The tests run on any machine, no sensor needed, a fake stands in for it:

```
python -m pytest tests -v
```

Values live in `config.py`, same names as the robot's config, so a module
written here drops into the robot code without an edit.
