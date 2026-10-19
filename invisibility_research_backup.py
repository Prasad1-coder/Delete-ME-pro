"""
invisibility.py  -  DeleteMe Pro engine (v3)

Gesture detection + person mask + cloak / ghost effects.
No GUI code here, so it can be used with OpenCV windows or PySide6.

Slow laptop? Lower FRAME_W / FRAME_H (e.g. 640 x 360). Everything else scales.
"""

import math
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

CYAN_BGR = np.array([255, 255, 0], np.float32)
GHOST_BGR = np.array([255, 90, 200], np.float32)


# ==========================================
# GESTURE DETECTION
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

    def set_background(self, bg_bgr_u8):
        self.bg_full = bg_bgr_u8.astype(np.float32)
        small = cv2.resize(bg_bgr_u8, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        self.bg_small = small.astype(np.float32)
        self.gain = np.ones(3, np.float32)
        self.prev_smooth = None

    def reset_temporal(self):
        self.prev_smooth = None

    # ---------- mask ----------

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

    # ---------- rendering ----------

    def _bbox(self, mask, pad=24):
        cols = np.flatnonzero(mask.max(axis=0) > 0.02)
        if cols.size == 0:
            return None
        rows = np.flatnonzero(mask.max(axis=1) > 0.02)
        x0, x1 = max(0, cols[0] - pad), min(FRAME_W, cols[-1] + pad + 1)
        y0, y1 = max(0, rows[0] - pad), min(FRAME_H, rows[-1] + pad + 1)
        return x0, y0, x1, y1

    def _refract(self, bg, ring, t, amp, x0, y0):
        """Heat-haze distortion of the background near the silhouette edge.
        Works on an ROI; x0/y0 keep the wave phase stable in screen space."""
        hr, wr = ring.shape
        half = cv2.resize(ring, (max(2, wr // 2), max(2, hr // 2)), interpolation=cv2.INTER_AREA)
        half = cv2.GaussianBlur(half, (0, 0), 3.5)
        band = np.clip(cv2.resize(half, (wr, hr)) * 3.0, 0.0, 1.0)
        sy = np.sin(self._ys[y0:y0 + hr] * 0.09 + t * 7.0)[:, None]
        sx = np.cos(self._xs[x0:x0 + wr] * 0.09 + t * 5.0)[None, :]
        mx = (self._gx[:hr, :wr] + amp * band * sy).astype(np.float32)
        my = (self._gy[:hr, :wr] + amp * band * sx).astype(np.float32)
        return cv2.remap(bg, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def composite(self, frame, mask, mode, strength, t):
        box = self._bbox(mask)
        if box is None:
            return frame
        x0, y0, x1, y1 = box
        m = mask[y0:y1, x0:x1]
        fr = frame[y0:y1, x0:x1].astype(np.float32)
        bg = self.bg_full[y0:y1, x0:x1] * self.gain

        target = TARGET_STRENGTH[mode]
        transition = min(1.0, abs(target - strength) * 3.0)
        ring = np.clip(4.0 * m * (1.0 - m), 0.0, 1.0)
        front = None

        if mode == "GHOST":
            self._color = GHOST_BGR
            flicker = 1.0 + 0.05 * math.sin(t * 13.0)
            alpha = np.clip(m * (strength * flicker), 0.0, 1.0)
            tint = (m * (0.16 * min(1.0, strength / 0.55)))[:, :, None]
            fr = fr + (GHOST_BGR - fr) * tint
            glow = 0.6
        else:
            if mode == "INVISIBLE":
                self._color = CYAN_BGR
            noise = self._noise[y0:y1, x0:x1]
            d = np.clip((strength * (1.0 + DISSOLVE_SOFT) - noise) / DISSOLVE_SOFT, 0.0, 1.0)
            alpha = m * d              # clean dissolve: no glow, no border, no distortion
            glow = 0.0

        out = fr + (bg - fr) * alpha[:, :, None]

        if glow > 0.0:                 # only ghost mode keeps a soft rim light
            pulse = 0.85 + 0.15 * math.sin(t * 6.0)
            out += (ring * (glow * pulse * 0.5))[:, :, None] * self._color

        result = frame.copy()
        result[y0:y1, x0:x1] = cv2.convertScaleAbs(out)     # saturating uint8
        return result