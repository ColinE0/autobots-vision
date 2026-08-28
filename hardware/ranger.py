class Ranger:
    def __init__(self, cfg, _sensor=None):
        self.cfg = cfg
        if _sensor is not None:                 # tests inject a fake here
            self._s = _sensor
        else:
            import VL53L0X                       # VL53L0X_rasp_python binding
            self._s = VL53L0X.VL53L0X(i2c_bus=cfg.RANGE_I2C_BUS,
                                      i2c_address=cfg.RANGE_I2C_ADDR)
            self._s.open()
            # Accuracy mode sets the timing budget; it must fit inside the
            # RANGE_POLL_HZ period (50 ms at 20 Hz). See config.RANGE_MODE.
            self._s.start_ranging(cfg.RANGE_MODE)
        self._alpha = cfg.RANGE_EMA_ALPHA
        self._period = 1.0 / cfg.RANGE_POLL_HZ
        self._next = 0.0
        self._d = None              # filtered metres
        self._last_valid = None     # time of the last valid read

    def _read_raw_m(self):
        """One sensor read in metres, or None if invalid. Tests override this."""
        try:
            mm = self._s.get_distance()
        except Exception:
            return None
        if mm is None or mm <= 0:
            return None
        return mm / 1000.0

    def poll(self, now):
        if now < self._next:
            return
        self._next = now + self._period
        d = self._read_raw_m()
        if d is None:
            return                              # hold the filter, freshness ages
        # ignore readings past the sensor's trustworthy range as no-target
        if d > self.cfg.RANGE_MAX_M:
            d = self.cfg.RANGE_MAX_M
        self._d = d if self._d is None else self._alpha * d + (1 - self._alpha) * self._d
        self._last_valid = now

    def distance_m(self):
        return self._d

    def fresh(self, now):
        return (self._last_valid is not None
                and now - self._last_valid <= self.cfg.RANGE_STALE_S)

    def close(self):
        for fn in ('stop_ranging', 'close'):
            try:
                getattr(self._s, fn)()
            except Exception:
                pass
