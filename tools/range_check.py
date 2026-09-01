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
Readings are appended, timestamped, to range_check.log in the repo folder
(on zone changes and every 2 s), so the run leaves a record.
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
from hardware.ranger import Ranger
from tools.sessionlog import SessionLog


def main():
    r = Ranger(config)
    log = SessionLog('range_check',
                     f"slow={config.RANGE_SLOW_M} hold={config.RANGE_HOLD_M} "
                     f"addr=0x{config.RANGE_I2C_ADDR:02x}")
    print("Ctrl+C to quit. SLOW / HOLD are the pilot's lines.\n")
    last_tag, t_last = None, 0.0
    try:
        while True:
            now = time.monotonic()
            r.poll(now)
            d = r.distance_m()
            if d is None or not r.fresh(now):
                row = tag = 'no read'
            else:
                tag = ('HOLD ' if d <= config.RANGE_HOLD_M else
                       'SLOW ' if d <= config.RANGE_SLOW_M else 'clear')
                row = f"{d:5.3f} m  {tag}"
            if tag != last_tag or now - t_last >= 2.0:   # zone changes + heartbeat
                log.line(row, echo=False)
                last_tag, t_last = tag, now
            print(f"\r{row}      ", end='', flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()
    finally:
        r.close()
        log.close()


if __name__ == '__main__':
    main()
