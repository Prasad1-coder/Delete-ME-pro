"""
invisibility.py  -  DeleteMe Pro engine (v4, visual upgrade)

Gesture detection + person mask + cloak / ghost effects.
No GUI code here, so it can be used with OpenCV windows or PySide6.

v4: gesture / mask / background logic is UNCHANGED. Only the rendering was
upgraded (composite + helpers). All effects run on the person's bounding box
only, so cost scales with the person size, not the full frame.

Slow laptop? Lower FRAME_W / FRAME_H (e.g. 640 x 360), or set FX_CHROMA_PX = 0
and FX_SPARKS = 0.0. Every effect below can be switched off from CONFIG.
"""

import math
import mediapipe as mp
import time
from collections import Counter, deque

import cv2
import numpy as np

# ==========================================
# CONFIG
# ==========================================

CAPTURE_W, CAPTURE_H = 1280, 720          # requested from the camera
FRAME_W, FRAME_H = 960, 540               # display + compositing size (16:9)
PROC_W, PROC_H = FRAME_W // 2, FRAME_H // 2   # size used by AI models / mask

COUNTDOWN_SEC = 7
BG_SAMPLES = 30
MIN_BLOB_RATIO = 0.01

TARGET_STRENGTH = {"NORMAL": 0.0, "INVISIBLE": 1.0, "GHOST": 0.55}
EASE = 0.15

# gestures
UP_ANGLE = 150          # finger straight above this angle
CURL_ANGLE = 130        # finger curled below this angle
HISTORY_LEN = 8
REQUIRED_VOTES = 6
MANUAL_LOCK_SEC = 2.5

# mask
MASK_EMA = 0.5
MASK_LOW, MASK_RANGE = 0.35, 0.30
DIFF_LOW, DIFF_HIGH = 30.0, 55.0          # background-difference evidence
DISSOLVE_SOFT = 0.25

# ---- visual effects (set to 0 to disable) ----
FX_REFRACT_AMP = 7.0      # heat-haze / glass distortion at the silhouette (pixels)
FX_CHROMA_PX = 2          # RGB split at the silhouette edge (pixels)
FX_FRONT_GLOW = 1.0       # glowing energy line while dissolving in / out
FX_SPARKS = 0.7           # glitter particles on the dissolve line
FX_RIM_INVIS = 0.14       # faint rim light around the invisible silhouette
FX_GHOST_RIM = 0.45       # ghost rim light strength
FX_GHOST_TRAIL = 0.86     # ghost motion echo decay (0 = off, 0.9 = long trail)
FX_GHOST_SCAN = 0.12      # ghost scanline strength

CYAN_BGR = np.array([255, 255, 0], np.float32)
GHOST_BGR = np.array([255, 90, 200], np.float32)
WHITE_BGR = np.array([255, 255, 255], np.float32)


def _m3(a):
    """2-D float mask -> 3-channel (fast, for cv2 arithmetic)."""
    return cv2.merge((a, a, a))


def _sc(v, k=1.0):
    """BGR numpy colour -> cv2 scalar."""
    return (float(v[0]) * k, float(v[1]) * k, float(v[2]) * k, 0.0)


# ==========================================
# GESTURE DETECTION  (unchanged)
# ==========================================

def _xyz(p):
    return np.array([p.x, p.y, p.z], np.float32)


def joint_angle(a, b, c):
    ba, bc = _xyz(a) - _xyz(b), _xyz(c) - _xyz(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def dist(a, b):
    return float(np.linalg.norm(_xyz(a) - _xyz(b)))


FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]


def _finger_state(lm, mcp, pip, tip):
    """+1 = clearly straight, -1 = clearly curled, 0 = in between."""
    angle = joint_angle(lm[mcp], lm[pip], lm[tip])
    if angle > UP_ANGLE and dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]):
        return 1
    if angle < CURL_ANGLE:
        return -1
    return 0


def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    s = [_finger_state(lm, *f) for f in FINGERS]      # index, middle, ring, pinky

    if all(v == -1 for v in s):
        return "FIST"
    if all(v == 1 for v in s):
        return "OPEN PALM"
    if s[0] == 1 and s[1] == 1 and s[2] == -1 and s[3] == -1:
        return "PEACE"
    return "UNKNOWN"          # half-curled fingers never trigger a mode


class GestureStabilizer:
    """Majority vote over recent frames. The mode only changes on a stable,
    valid gesture. UNKNOWN / NO HAND keep the current mode (sticky)."""

    GESTURE_TO_MODE = {"FIST": "INVISIBLE", "OPEN PALM": "NORMAL", "PEACE": "GHOST"}

    def __init__(self):
        self.history = deque(maxlen=HISTORY_LEN)
        self.mode = "NORMAL"
        self.lock_until = 0.0

    @property
    def locked(self):
        return time.time() < self.lock_until

    def force_mode(self, mode):
        """Keyboard override: gestures are ignored for a short time."""
        self.mode = mode
        self.history.clear()
        self.lock_until = time.time() + MANUAL_LOCK_SEC

    def update(self, gesture):
        if self.locked:
            return self.mode
        self.history.append(gesture)
        top, votes = Counter(self.history).most_common(1)[0]
        if top in self.GESTURE_TO_MODE and votes >= REQUIRED_VOTES:
            self.mode = self.GESTURE_TO_MODE[top]
        return self.mode


# ==========================================
# CLOAK ENGINE
# ==========================================

class CloakEngine:
    """Builds a clean person mask and renders the invisibility / ghost look."""

    def __init__(self):
        self.bg_full = None
        self.bg_small = None
        self.gain = np.ones(3, np.float32)
        self.prev_smooth = None
        self._color = CYAN_BGR

        self.min_area = int(PROC_W * PROC_H * MIN_BLOB_RATIO)
        self.k_region = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # cached grids / noise
        xs = np.arange(FRAME_W, dtype=np.float32)
        ys = np.arange(FRAME_H, dtype=np.float32)
        self._xs, self._ys = xs, ys
        self._gx, self._gy = np.meshgrid(xs, ys)
        self._noise = self._make_noise()
        self._grain = self._make_grain()
        self._rng = np.random.default_rng()

        # ghost motion-echo buffer
        self.trail = np.zeros((FRAME_H, FRAME_W), np.float32)
        self._trail_live = False

    # ---------- setup ----------

    def _make_noise(self):
        rng = np.random.default_rng(7)
        small = rng.random((FRAME_H // 10, FRAME_W // 10)).astype(np.float32)
        n = cv2.resize(small, (FRAME_W, FRAME_H), interpolation=cv2.INTER_CUBIC)
        n = cv2.GaussianBlur(n, (0, 0), 3)
        n = (n - n.min()) / (n.max() - n.min() + 1e-6)
        sweep = np.linspace(0.0, 0.35, FRAME_H, dtype=np.float32)[:, None]
        n = n + sweep                                   # dissolve sweeps top -> bottom
        return (n - n.min()) / (n.max() - n.min() + 1e-6)

    def _make_grain(self):
        """Sparse glitter map (2x frame size). Sliced at a random offset every
        frame, so sparks appear to flicker for free."""
        rng = np.random.default_rng(11)
        g = rng.random((FRAME_H * 2, FRAME_W * 2)).astype(np.float32)
        dots = ((g > 0.987) * rng.random(g.shape).astype(np.float32)).astype(np.float32)
        dots = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return np.clip(dots * 1.6, 0.0, 1.0)

    def set_background(self, bg_bgr_u8):
        self.bg_full = bg_bgr_u8.astype(np.float32)
        small = cv2.resize(bg_bgr_u8, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        self.bg_small = small.astype(np.float32)
        self.gain = np.ones(3, np.float32)
        self.prev_smooth = None
        self.trail[:] = 0
        self._trail_live = False

    def reset_temporal(self):
        self.prev_smooth = None
        self.trail[:] = 0
        self._trail_live = False

    # ---------- mask  (unchanged) ----------

    def _update_gain(self, fs, raw):
        """Match background brightness/colour to the live frame."""
        outside = (raw < 0.05)[::2, ::2]
        if outside.sum() > 1500:
            f = fs[::2, ::2][outside].mean(axis=0)
            b = self.bg_small[::2, ::2][outside].mean(axis=0)
            g = np.clip(f / (b + 1e-3), 0.7, 1.4).astype(np.float32)
            self.gain = 0.9 * self.gain + 0.1 * g

    def update_mask(self, frame_small, raw):
        """frame_small: BGR uint8 (PROC size). raw: selfie mask (PROC size).
        Returns a feathered float mask at FRAME size."""
        fs = frame_small.astype(np.float32)
        self._update_gain(fs, raw)

        smooth = raw if self.prev_smooth is None \
            else MASK_EMA * self.prev_smooth + (1 - MASK_EMA) * raw
        self.prev_smooth = smooth
        soft = np.clip((smooth - MASK_LOW) / MASK_RANGE, 0.0, 1.0)

        # extra evidence: pixels that differ from the clean background,
        # but only near the AI mask (rescues hair, fingers, loose clothes)
        bg_adj = np.clip(self.bg_small * self.gain, 0, 255)
        diff = np.abs(fs - bg_adj).max(axis=2)
        diff = cv2.GaussianBlur(diff, (0, 0), 2)
        diff_soft = np.clip((diff - DIFF_LOW) / (DIFF_HIGH - DIFF_LOW), 0.0, 1.0)
        near = cv2.dilate((soft > 0.3).astype(np.uint8), self.k_region)
        near = cv2.GaussianBlur(near.astype(np.float32), (0, 0), 3)

        combined = np.maximum(soft, diff_soft * near)

        # fill holes, drop small blobs
        binary = (combined > 0.5).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.k_close)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        valid = np.zeros(n, dtype=bool)
        valid[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_area
        keep = valid[labels].astype(np.float32)
        combined = np.maximum(combined * keep, binary.astype(np.float32) * keep * 0.95)

        combined = cv2.dilate(combined, self.k_edge)          # hide the halo
        combined = cv2.GaussianBlur(combined, (0, 0), 1.6)     # feather (small res = cheap)
        mask = cv2.resize(combined, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)

    # ---------- rendering helpers ----------

    def _bbox(self, mask, pad=24):
        cols = np.flatnonzero(mask.max(axis=0) > 0.02)
        if cols.size == 0:
            return None
        rows = np.flatnonzero(mask.max(axis=1) > 0.02)
        x0, x1 = max(0, cols[0] - pad), min(FRAME_W, cols[-1] + pad + 1)
        y0, y1 = max(0, rows[0] - pad), min(FRAME_H, rows[-1] + pad + 1)
        return x0, y0, x1, y1

    @staticmethod
    def _edge_band(m):
        """Soft band centred on the silhouette edge (1.0 on the edge, 0 far
        away). Computed at half resolution, so it is cheap."""
        hr, wr = m.shape
        half = cv2.resize(m, (max(2, wr // 2), max(2, hr // 2)), interpolation=cv2.INTER_AREA)
        half = cv2.GaussianBlur(half, (0, 0), 3.0)
        band = np.clip(4.0 * half * (1.0 - half), 0.0, 1.0)
        return cv2.resize(band, (wr, hr), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _mix(img, color, k):
        """img + (color - img) * k   (k: 2-D float mask)"""
        c = np.empty_like(img)
        c[:] = color
        return cv2.add(img, cv2.multiply(cv2.subtract(c, img), _m3(k)))

    def _refract(self, img, band, t, amp, x0, y0):
        """Heat-haze / glass distortion applied near the silhouette edge.
        Works on the ROI; x0/y0 keep the wave phase stable in screen space."""
        hr, wr = band.shape
        sy = (np.sin(self._ys[y0:y0 + hr] * 0.09 + t * 7.0)
              + 0.5 * np.sin(self._ys[y0:y0 + hr] * 0.23 - t * 11.0))[:, None]
        sx = (np.cos(self._xs[x0:x0 + wr] * 0.09 + t * 5.0)
              + 0.5 * np.cos(self._xs[x0:x0 + wr] * 0.21 + t * 9.0))[None, :]
        mx = (self._gx[:hr, :wr] + (amp * 0.7) * band * sy).astype(np.float32)
        my = (self._gy[:hr, :wr] + (amp * 0.7) * band * sx).astype(np.float32)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ---------- rendering ----------

    def composite(self, frame, mask, mode, strength, t):
        ghost = mode == "GHOST"

        # ghost motion echo: decays when ghost mode is off
        trail_on = ghost and FX_GHOST_TRAIL > 0
        if trail_on:
            np.maximum(self.trail * FX_GHOST_TRAIL, mask, out=self.trail)
            self._trail_live = True
        elif self._trail_live:
            self.trail *= 0.6
            if self.trail.max() < 0.02:
                self.trail[:] = 0
                self._trail_live = False

        if mode == "NORMAL" and strength < 0.003 and not self._trail_live:
            return frame

        cover = np.maximum(mask, self.trail) if self._trail_live else mask
        box = self._bbox(cover, pad=28)
        if box is None:
            return frame
        x0, y0, x1, y1 = box
        h, w = y1 - y0, x1 - x0

        m = mask[y0:y1, x0:x1]
        fr = frame[y0:y1, x0:x1].astype(np.float32)
        bg = cv2.multiply(self.bg_full[y0:y1, x0:x1], _sc(self.gain))
        band = self._edge_band(m)
        vis = min(1.0, strength)

        if ghost:
            self._color = GHOST_BGR
            flicker = 1.0 + 0.05 * math.sin(t * 13.0)
            alpha = np.clip(m * (strength * flicker), 0.0, 1.0)

            if FX_GHOST_SCAN > 0:       # slow scanlines rolling over the body
                scan = 0.5 + 0.5 * np.sin(self._ys[y0:y1] * 1.7 - t * 9.0)
                alpha = alpha * (1.0 - FX_GHOST_SCAN * scan)[:, None]

            tint = m * (0.16 * min(1.0, strength / 0.55))
            fr = self._mix(fr, GHOST_BGR, tint)

            if trail_on:                # motion echo: coloured smear behind movement
                echo = np.clip(self.trail[y0:y1, x0:x1] - m, 0.0, 1.0)
                fr = self._mix(fr, GHOST_BGR, echo * 0.40)

            out = cv2.add(fr, cv2.multiply(cv2.subtract(bg, fr), _m3(alpha)))
            rim_k = FX_GHOST_RIM * (0.85 + 0.15 * math.sin(t * 6.0))
        else:
            if mode == "INVISIBLE":
                self._color = CYAN_BGR
            noise = self._noise[y0:y1, x0:x1]
            d = np.clip((strength * (1.0 + DISSOLVE_SOFT) - noise) / DISSOLVE_SOFT, 0.0, 1.0)
            alpha = m * d
            out = cv2.add(fr, cv2.multiply(cv2.subtract(bg, fr), _m3(alpha)))
            rim_k = FX_RIM_INVIS

            # energy line + sparks, only visible while dissolving in / out
            if FX_FRONT_GLOW > 0:
                front = 4.0 * d * (1.0 - d) * m
                if front.max() > 0.02:
                    energy = self._color * 0.55 + WHITE_BGR * 0.30
                    out = cv2.add(out, cv2.multiply(_m3(front * FX_FRONT_GLOW), _sc(energy)))
                    if FX_SPARKS > 0:
                        oy = int(self._rng.integers(0, FRAME_H))
                        ox = int(self._rng.integers(0, FRAME_W))
                        sp = self._grain[oy:oy + h, ox:ox + w] * front * FX_SPARKS
                        out = cv2.add(out, cv2.multiply(_m3(sp), _sc(WHITE_BGR)))

        # shared silhouette effects ------------------------------------
        amp = FX_REFRACT_AMP * vis * (0.4 if ghost else 1.0)
        if amp > 0.3:
            out = self._refract(out, band, t, amp, x0, y0)

        if FX_CHROMA_PX > 0 and vis > 0.05:
            k = FX_CHROMA_PX
            wgt = band * vis
            b_shift = np.roll(out[:, :, 0], k, axis=1)
            r_shift = np.roll(out[:, :, 2], -k, axis=1)
            out[:, :, 0] += (b_shift - out[:, :, 0]) * wgt
            out[:, :, 2] += (r_shift - out[:, :, 2]) * wgt

        if rim_k > 0 and vis > 0.02:
            out = cv2.add(out, cv2.multiply(_m3(band * (rim_k * vis)), _sc(self._color)))

        result = frame.copy()
        result[y0:y1, x0:x1] = cv2.convertScaleAbs(out)     # saturating uint8
        return result