# autobots-vision

Sensor code for our senior design robot (AutoBots, Team 2.09). This is where we
put the camera work and the forward rangefinder so we can both pull it, run it,
and not overwrite each other. The rest of the robot code lives separately.

## Layout

```
hardware/   sensor drivers. So far the VL53L0X time-of-flight rangefinder
vision/     the camera modules, once we have them in here
tests/      whatever we can run without the robot
tools/      bench scripts. tools/range_check.py is the live ToF readout
```

The tests run on any machine, no sensor needed, a fake stands in for it:

```
python -m pytest tests -v
```

Values live in `config.py`, same names as the robot's config, so a module
written here drops into the robot code without an edit.
