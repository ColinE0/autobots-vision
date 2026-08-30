# autobots-vision

Sensor code for our senior design robot (AutoBots, Team 2.09). This is where we
put the camera work and the forward rangefinder so we can both pull it, run it,
and not overwrite each other. The rest of the robot code lives separately.

## Layout

```
hardware/   sensor drivers. So far the VL53L0X time-of-flight rangefinder
vision/     the camera modules: stop sign + traffic light detectors, and
            detector_geometric.py, the wrapper that speaks the robot's
            detect() -> [Detection(label, area_frac)] contract
tests/      whatever we can run without the robot
tools/      bench scripts. tools/range_check.py is the live ToF readout,
            tools/detect_preview.py is the live camera detection preview
```

The tests run on any machine, no sensor needed, a fake stands in for it:

```
python -m pytest tests -v
```

Values live in `config.py`, same names as the robot's config, so a module
written here drops into the robot code without an edit.
