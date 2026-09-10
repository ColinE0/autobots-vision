"""
HSV probe: the numbers the classical detector judges, per colour, per frame.

Pure functions over one BGR frame; no camera, no printing. tools/probe_hsv.py
is the live wrapper, tests/test_probe.py pins the arithmetic. The point is to
read WHY a lamp or sign is or is not being taken, in the detector's own units,
instead of guessing at LIGHT_S_MIN / LIGHT_V_MIN / LAMP_GLOW by eye.

Two readings per frame:

  centre:  median H, S, V of a small patch at the frame centre, so a prop
           held in the crosshair can be read directly ("that red lens is
           S 140 V 95: under the lamp floor, over the sign floor").
  blobs:   for each colour, the LARGEST blob the detector's own mask and
           shape gates would consider, with the glow statistics _glows()
           computes and the first gate it fails, if any.

The blob path mirrors Detector.detect() step for step (resize to
DETECT_PROC_WIDTH, floor cut, per-colour band, 3x3 open, area and shape
gates). Keep the two in step: a probe that measures a different blob than
the detector judges is worse than none.
"""
import cv2
import numpy as np

from vision.detector import _light_bands, _prep, _mask, _clipped_core, WHITE_S_MAX

COLORS = ('red', 'yellow', 'green')


def centre_hsv(frame_bgr, patch=20):
    """Median (H, S, V) of a patch x patch box at the frame centre."""
    h, w = frame_bgr.shape[:2]
    half = max(1, patch // 2)
    cy, cx = h // 2, w // 2
    box = frame_bgr[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med = np.median(hsv, axis=0)
    return tuple(int(round(x)) for x in med)


def glow_stats(v_vals, v_floor, gate, min_spread=0, min_pixels=16):
    """The quantities _glows() judges, plus the first gate that fails.

    Returns a dict: n, bright (count at/above bright level), bright_frac,
    mean, p10, p90, spread, and fail (None when the blob glows).
    """
    mean_min, peak_min, bright_frac, bright_v = gate
    out = {'n': int(v_vals.size), 'bright': 0, 'bright_frac': 0.0,
           'mean': 0.0, 'p10': 0.0, 'p90': 0.0, 'spread': 0.0, 'fail': None}
    if v_vals.size == 0:
        out['fail'] = 'no pixels'
        return out
    bright = int(np.count_nonzero(v_vals >= max(v_floor, bright_v)))
    out['bright'] = bright
    out['bright_frac'] = bright / float(v_vals.size)
    v = v_vals[v_vals >= v_floor]
    if v.size:
        p10, p90 = np.percentile(v, (10, 90))
        out['mean'] = float(v.mean())
        out['p10'], out['p90'] = float(p10), float(p90)
        out['spread'] = float(p90 - p10)
    if bright < min_pixels:
        out['fail'] = f'bright pixels {bright} < {min_pixels}'
    elif out['bright_frac'] < bright_frac:
        out['fail'] = f'bright share {out["bright_frac"]:.2f} < {bright_frac:.2f}'
    elif v.size == 0:
        out['fail'] = f'nothing at/above V {v_floor}'
    elif min_spread > 0 and out['spread'] < min_spread:
        out['fail'] = f'spread {out["spread"]:.0f} < {min_spread}'
    elif out['mean'] < mean_min:
        out['fail'] = f'mean V {out["mean"]:.0f} < {mean_min}'
    elif out['p90'] < peak_min:
        out['fail'] = f'p90 V {out["p90"]:.0f} < {peak_min}'
    return out


class Probe:
    """Per-colour largest-blob statistics, built the way Detector builds them."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._kernel = np.ones((3, 3), np.uint8)
        lamp = _light_bands(cfg.LIGHT_S_MIN, cfg.LIGHT_V_MIN)
        sign = _light_bands(cfg.LIGHT_S_MIN, cfg.SIGN_V_MIN)
        self._bands = {'red': _prep(sign['red']),
                       'yellow': _prep(lamp['yellow']),
                       'green': _prep(lamp['green'])}
        self._glow = {k: tuple(v) for k, v in cfg.LAMP_GLOW.items()}
        self._core = (cfg.LAMP_CORE_MIN_FRAC, tuple(cfg.LAMP_CORE_ASPECT),
                      cfg.LAMP_CORE_MIN_EXTENT, cfg.LAMP_CORE_CENTER_TOL)

    def measure(self, frame_bgr):
        """Dict colour -> stats for the largest candidate blob, or None.

        Each stats dict carries: area_frac, s_mean, v_mean, and the
        glow_stats() fields (n, bright, bright_frac, mean, p10, p90,
        spread, fail), core (True when the blob would pass the detector's
        clipped-core test, the blown-out-lamp path that runs after glow
        fails), and core_px (white pixels at/above LAMP_CORE_V_MIN inside
        the blob). 'shape' is the reason a blob was skipped before the
        glow test, when the largest mask region failed the shape gate and
        nothing else qualified.
        """
        cfg = self.cfg
        h, w = frame_bgr.shape[:2]
        if w != cfg.DETECT_PROC_WIDTH:
            scale = cfg.DETECT_PROC_WIDTH / float(w)
            frame_bgr = cv2.resize(frame_bgr,
                                   (cfg.DETECT_PROC_WIDTH, max(1, int(h * scale))))
        sh, sw = frame_bgr.shape[:2]
        frame_area = float(sh * sw)
        cut = int(sh * (1.0 - cfg.DETECT_IGNORE_BOTTOM_FRAC))
        hsv = cv2.cvtColor(frame_bgr[:cut], cv2.COLOR_BGR2HSV)
        sch, vch = hsv[:, :, 1], hsv[:, :, 2]
        spread = getattr(cfg, 'LAMP_MIN_V_SPREAD', 0)
        min_px = getattr(cfg, 'LAMP_MIN_BRIGHT_PIXELS', 16)

        result = {}
        for color in COLORS:
            mask = cv2.morphologyEx(_mask(hsv, self._bands[color]),
                                    cv2.MORPH_OPEN, self._kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            best, best_frac, skipped = None, 0.0, None
            for c in contours:
                area = cv2.contourArea(c)
                frac = area / frame_area
                if frac < cfg.DETECT_MIN_AREA_FRAC:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                extent = area / float(bw * bh)
                aspect = bw / float(bh)
                if extent < 0.45 or not (0.4 < aspect < 2.5):
                    if skipped is None or frac > skipped[0]:
                        skipped = (frac, f'shape extent {extent:.2f} aspect {aspect:.2f}')
                    continue
                if frac > best_frac:
                    best, best_frac = (c, x, y, bw, bh), frac
            if best is None:
                result[color] = ({'area_frac': skipped[0], 'shape': skipped[1]}
                                 if skipped else None)
                continue
            c, x, y, bw, bh = best
            box = (slice(y, y + bh), slice(x, x + bw))
            sel = mask[box] > 0
            vals = vch[box][sel]
            stats = glow_stats(vals, cfg.LIGHT_V_MIN, self._glow[color],
                               spread, min_px)
            stats['area_frac'] = best_frac
            stats['s_mean'] = float(sch[box][sel].mean())
            stats['v_mean'] = float(vals.mean())
            stats['shape'] = None
            inside = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(inside, [c - (x, y)], -1, 255, cv2.FILLED)
            core = ((sch[box] < WHITE_S_MAX) & (vch[box] >= cfg.LAMP_CORE_V_MIN)
                    & (inside > 0))
            stats['core_px'] = int(np.count_nonzero(core))
            stats['core'] = bool(_clipped_core(core.astype(np.uint8) * 255,
                                               *self._core))
            result[color] = stats
        return result


def format_line(centre, blobs):
    """One terminal line per frame: centre patch, then each colour's blob."""
    parts = [f"centre H{centre[0]:3d} S{centre[1]:3d} V{centre[2]:3d}"]
    for color in COLORS:
        st = blobs.get(color)
        if st is None:
            parts.append(f"{color[0].upper()}: -")
            continue
        if st.get('shape'):
            parts.append(f"{color[0].upper()}: {st['area_frac']*100:.1f}% {st['shape']}")
            continue
        if st['fail'] is None:
            verdict = 'GLOWS'
        elif st['core']:
            verdict = f"CORE ({st['fail']})"
        else:
            verdict = st['fail']
        parts.append(f"{color[0].upper()}: {st['area_frac']*100:.1f}% "
                     f"S{st['s_mean']:.0f} V{st['v_mean']:.0f} "
                     f"p90 {st['p90']:.0f} br {st['bright_frac']:.2f}/{st['bright']} "
                     f"core {st['core_px']} [{verdict}]")
    return ' | '.join(parts)
