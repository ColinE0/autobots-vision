"""
Live rangefinder readout. Run it on the Pi with the VL53L0X on the I2C bus.

    cd autobots-vision && python3 -m tools.range_check

Walk a target (a hand, a box, the other bot) in and out of the cone and
confirm:
  1. The distance falls smoothly as the target comes in, no jumps to zero.
  2. An empty aisle reads at the 2.00 m clamp, not some fixed short number.
     A fixed short number means the FLOOR is in the cone: raise the mount or
     tilt it up a degree, do not tune it away in software.
  3. SLOW and HOLD light up where you expect them physically. Those are the
     lines the pilot slows and stops on.
Measure the distances with a tape the first time. If the reading is off by a
constant, say so before trusting anything downstream of it.
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
from hardware.ranger import Ranger


def main():
    r = Ranger(config)
    print("Ctrl+C to quit. SLOW / HOLD are the pilot's lines.\n")
    try:
        while True:
            now = time.monotonic()
            r.poll(now)
            d = r.distance_m()
            if d is None or not r.fresh(now):
                row = 'no read'
            else:
                tag = ('HOLD ' if d <= config.RANGE_HOLD_M else
                       'SLOW ' if d <= config.RANGE_SLOW_M else 'clear')
                row = f"{d:5.3f} m  {tag}"
            print(f"\r{row}      ", end='', flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()
    finally:
        r.close()


if __name__ == '__main__':
    main()
