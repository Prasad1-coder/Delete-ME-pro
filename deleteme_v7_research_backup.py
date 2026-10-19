import os
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
)

from invisibility import (
    CAPTURE_H,
    CAPTURE_W,
    COUNTDOWN_SEC,
    BG_SAMPLES,
    EASE,
    FRAME_H,
    FRAME_W,
    PROC_H,
    PROC_W,
    TARGET_STRENGTH,
    CloakEngine,
    GestureStabilizer,
    detect_gesture,
)


WINDOW_TITLE = "DELETEME PRO // CYBER VISION"

HUD_SCALE = 1.0

WHITE = (240, 240, 240)
MUTED = (160, 165, 170)
DIM = (110, 115, 120)
DARK = (20, 22, 24)
GREEN = (125, 205, 135)
CYAN = (235, 195, 105)
VIOLET = (215, 150, 200)

MODE_COLORS = {
    "NORMAL": GREEN,
    "INVISIBLE": CYAN,
    "GHOST": VIOLET,
}

STATUS_TEXT = {
    "NORMAL": "READY",
    "INVISIBLE": "CLOAK ACTIVE",
    "GHOST": "GHOST ACTIVE",
}


# =========================================================
# HUD HELPERS
# =========================================================

def qc(bgr, alpha=255):
    return QColor(bgr[2], bgr[1], bgr[0], alpha)


def _font(px, bold=False, spacing=0.0):
    font = QFont()

    try:
        font.setFamilies(
            ["Segoe UI", "Inter", "Helvetica Neue", "Arial"]
        )
    except Exception:
        font.setFamily("Segoe UI")

    font.setPixelSize(max(7, int(round(px))))
    font.setWeight(
        QFont.Weight.DemiBold if bold else QFont.Weight.Normal
    )

    if spacing:
        font.setLetterSpacing(
            QFont.SpacingType.AbsoluteSpacing,
            spacing,
        )

    return font


def _width(text, font):
    return QFontMetricsF(font).horizontalAdvance(text)


def _draw(painter, text, x, y, font, color):
    painter.setFont(font)
    painter.setPen(qc(color))
    painter.drawText(QPointF(x, y), text)


def paint_hud(painter, width, height, state):
    """Draw a crisp cyberpunk HUD over the camera image."""

    painter.setRenderHint(
        QPainter.RenderHint.Antialiasing,
        True,
    )
    painter.setRenderHint(
        QPainter.RenderHint.TextAntialiasing,
        True,
    )

    u = (width / 1280.0) * HUD_SCALE
    margin = 16 * u

    mode = state.get("mode", "NORMAL")
    accent = MODE_COLORS.get(mode, GREEN)

    def panel(x, y, panel_width, panel_height, alpha=178):
        painter.setPen(
            QPen(
                QColor(255, 255, 255, 28),
                max(1.0, 0.8 * u),
            )
        )

        painter.setBrush(
            QColor(DARK[2], DARK[1], DARK[0], alpha)
        )

        painter.drawRoundedRect(
            QRectF(x, y, panel_width, panel_height),
            4 * u,
            4 * u,
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qc(accent))

        painter.drawRoundedRect(
            QRectF(
                x,
                y + 6 * u,
                2.6 * u,
                panel_height - 12 * u,
            ),
            1.3 * u,
            1.3 * u,
        )

    # -----------------------------------------------------
    # MAIN HUD
    # -----------------------------------------------------

    if state.get("hud", True):

        # Top-left branding
        bx, by = margin, margin
        bw, bh = 210 * u, 48 * u

        panel(bx, by, bw, bh)

        _draw(
            painter,
            "DELETEME PRO",
            bx + 15 * u,
            by + 21 * u,
            _font(14 * u, True, 0.6),
            WHITE,
        )

        _draw(
            painter,
            "COMPUTER VISION",
            bx + 15 * u,
            by + 37 * u,
            _font(8.5 * u, False, 1.2),
            MUTED,
        )

        _draw(
            painter,
            "1 NORMAL  2 CLOAK  3 GHOST",
            bx + 2 * u,
            by + bh + 17 * u,
            _font(8.5 * u, False, 0.4),
            DIM,
        )

        _draw(
            painter,
            "B BG  V REC  7 HD  8 FHD  S SHOT",
            bx + 2 * u,
            by + bh + 30 * u,
            _font(8.5 * u, False, 0.3),
            DIM,
        )

        # Top-right telemetry
        tw, th = 190 * u, 48 * u

        tx = width - margin - tw
        ty = margin

        panel(tx, ty, tw, th)

        _draw(
            painter,
            "FPS",
            tx + 15 * u,
            ty + 17 * u,
            _font(8 * u, False, 1.2),
            MUTED,
        )

        _draw(
            painter,
            "MODE",
            tx + 72 * u,
            ty + 17 * u,
            _font(8 * u, False, 1.2),
            MUTED,
        )

        _draw(
            painter,
            f"{state.get('fps', 0):.0f}",
            tx + 15 * u,
            ty + 35 * u,
            _font(15 * u, True),
            WHITE,
        )

        _draw(
            painter,
            mode,
            tx + 72 * u,
            ty + 34 * u,
            _font(12 * u, True, 0.5),
            accent,
        )

        # Effect strength bar
        bar_x = tx + 15 * u
        bar_y = ty + th - 8 * u
        bar_width = tw - 30 * u
        bar_height = 2.6 * u

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 40))

        painter.drawRoundedRect(
            QRectF(
                bar_x,
                bar_y,
                bar_width,
                bar_height,
            ),
            bar_height / 2,
            bar_height / 2,
        )

        max_strength = max(TARGET_STRENGTH.values())
        strength = state.get("strength", 0.0)

        fraction = min(
            1.0,
            strength / max(max_strength, 0.001),
        )

        if fraction > 0.01:
            painter.setBrush(qc(accent))

            painter.drawRoundedRect(
                QRectF(
                    bar_x,
                    bar_y,
                    bar_width * fraction,
                    bar_height,
                ),
                bar_height / 2,
                bar_height / 2,
            )

        # Bottom-left gesture indicator
        gesture = state.get("gesture", "---")

        gh = 32 * u
        gy = height - margin - gh

        label_font = _font(8 * u, False, 1.0)
        value_font = _font(12 * u, True, 0.4)

        label_width = _width("GESTURE", label_font)
        value_width = _width(gesture, value_font)

        gesture_width = (
            15 * u
            + label_width
            + 14 * u
            + value_width
            + 15 * u
        )

        panel(margin, gy, gesture_width, gh)

        _draw(
            painter,
            "GESTURE",
            margin + 15 * u,
            gy + 20 * u,
            label_font,
            MUTED,
        )

        gesture_color = (
            DIM if gesture in ("NO HAND", "---") else WHITE
        )

        _draw(
            painter,
            gesture,
            margin + 15 * u + label_width + 14 * u,
            gy + 21 * u,
            value_font,
            gesture_color,
        )

        # Bottom-right mode status
        status = STATUS_TEXT.get(mode, "ONLINE")

        if state.get("manual", False):
            status += "   |   MANUAL"

        status_font = _font(10.5 * u, True, 0.6)
        status_width = _width(status, status_font) + 32 * u

        status_x = width - margin - status_width

        panel(status_x, gy, status_width, gh)

        _draw(
            painter,
            status,
            status_x + 16 * u,
            gy + 21 * u,
            status_font,
            accent,
        )

    # -----------------------------------------------------
    # RECORDING INDICATOR
    # -----------------------------------------------------

    if state.get("recording", False):

        elapsed = int(state.get("record_seconds", 0))
        minutes, seconds = divmod(elapsed, 60)

        quality = state.get("record_quality", "720p")

        rec_text = (
            f"REC  {minutes:02d}:{seconds:02d}  |  {quality}"
        )

        rec_font = _font(11 * u, True, 0.6)

        rec_width = _width(rec_text, rec_font) + 38 * u
        rec_height = 32 * u

        rec_x = (width - rec_width) / 2
        rec_y = 18 * u

        painter.setPen(
            QPen(
                QColor(255, 90, 90, 100),
                max(1.0, u),
            )
        )

        painter.setBrush(QColor(20, 22, 24, 220))

        painter.drawRoundedRect(
            QRectF(
                rec_x,
                rec_y,
                rec_width,
                rec_height,
            ),
            5 * u,
            5 * u,
        )

        painter.setPen(QColor(255, 70, 70))
        painter.setBrush(QColor(255, 65, 65))

        painter.drawEllipse(
            QRectF(
                rec_x + 12 * u,
                rec_y + 11 * u,
                8 * u,
                8 * u,
            )
        )

        _draw(
            painter,
            rec_text,
            rec_x + 27 * u,
            rec_y + 21 * u,
            rec_font,
            WHITE,
        )

    # -----------------------------------------------------
    # TOAST MESSAGE
    # -----------------------------------------------------

    toast = state.get("toast")

    if toast:

        toast_font = _font(11 * u, True, 0.5)

        toast_width = _width(toast, toast_font) + 34 * u
        toast_height = 30 * u

        toast_x = (width - toast_width) / 2
        toast_y = 58 * u

        painter.setPen(
            QPen(
                QColor(255, 255, 255, 28),
                max(1.0, 0.8 * u),
            )
        )

        painter.setBrush(QColor(20, 22, 24, 220))

        painter.drawRoundedRect(
            QRectF(
                toast_x,
                toast_y,
                toast_width,
                toast_height,
            ),
            5 * u,
            5 * u,
        )

        _draw(
            painter,
            toast,
            toast_x + 17 * u,
            toast_y + 20 * u,
            toast_font,
            WHITE,
        )

    # -----------------------------------------------------
    # BACKGROUND SCAN
    # -----------------------------------------------------

    scan = state.get("scan")

    if scan:

        remaining, progress, text = scan

        card_width = 400 * u
        card_height = 182 * u

        center_x = width / 2
        center_y = height / 2

        painter.setPen(
            QPen(
                QColor(255, 255, 255, 30),
                max(1.0, 0.8 * u),
            )
        )

        painter.setBrush(QColor(20, 22, 24, 218))

        painter.drawRoundedRect(
            QRectF(
                center_x - card_width / 2,
                center_y - card_height / 2,
                card_width,
                card_height,
            ),
            8 * u,
            8 * u,
        )

        ring = QRectF(
            center_x - 48 * u,
            center_y - card_height / 2 + 26 * u,
            96 * u,
            96 * u,
        )

        painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.setPen(
            QPen(
                QColor(255, 255, 255, 46),
                3.5 * u,
            )
        )

        painter.drawEllipse(ring)

        ring_pen = QPen(qc(CYAN), 3.5 * u)
        ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(ring_pen)

        painter.drawArc(
            ring,
            90 * 16,
            -int(360 * 16 * min(1.0, progress)),
        )

        painter.setFont(_font(38 * u))
        painter.setPen(qc(WHITE))

        painter.drawText(
            ring,
            int(Qt.AlignmentFlag.AlignCenter),
            str(remaining) if remaining > 0 else "...",
        )

        scan_font = _font(10.5 * u, False, 1.0)
        scan_width = _width(text, scan_font)

        _draw(
            painter,
            text,
            center_x - scan_width / 2,
            ring.bottom() + 32 * u,
            scan_font,
            MUTED,
        )


# =========================================================
# CAMERA WORKER
# =========================================================

class CameraWorker(QThread):

    frame_ready = Signal(QImage, object)
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()

        self.running = False
        self.camera = None

        self.engine = CloakEngine()
        self.stabilizer = GestureStabilizer()
        self.strength = 0.0

        self.command_lock = threading.Lock()
        self.manual_mode = None

        self.recapture_requested = threading.Event()
        self.screenshot_requested = threading.Event()

        self.show_hud = True
        self.show_landmarks = True

        self._toast = None

    # -----------------------------------------------------
    # HELPERS
    # -----------------------------------------------------

    def post_toast(self, text, seconds=2.0):
        self._toast = (text, time.time() + seconds)

    def _current_toast(self):
        if self._toast and time.time() < self._toast[1]:
            return self._toast[0]

        return None

    def open_camera(self):

        camera = None

        if sys.platform.startswith("win"):
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if camera is None or not camera.isOpened():

            if camera is not None:
                camera.release()

            camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            camera.release()

            raise RuntimeError(
                "Camera open nahi hua. Permissions aur dusre "
                "camera apps check karo."
            )

        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

        return camera

    def read_frame(self):

        if self.camera is None:
            return None

        success, frame = self.camera.read()

        if not success or frame is None:
            return None

        # Center-crop to the target aspect ratio.
        height, width = frame.shape[:2]
        target_ratio = FRAME_W / FRAME_H

        if width / height < target_ratio:

            new_height = int(width / target_ratio)

            y0 = (height - new_height) // 2

            frame = frame[y0:y0 + new_height]

        elif width / height > target_ratio:

            new_width = int(height * target_ratio)

            x0 = (width - new_width) // 2

            frame = frame[:, x0:x0 + new_width]

        if frame.shape[1] != FRAME_W:

            interpolation = (
                cv2.INTER_AREA
                if frame.shape[1] > FRAME_W
                else cv2.INTER_LINEAR
            )

            frame = cv2.resize(
                frame,
                (FRAME_W, FRAME_H),
                interpolation=interpolation,
            )

        return cv2.flip(frame, 1)

    def emit_frame(
        self,
        frame,
        mode="NORMAL",
        gesture="---",
        fps=0.0,
        scan=None,
    ):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

        state = {
            "hud": self.show_hud,
            "mode": mode,
            "gesture": gesture,
            "fps": fps,
            "strength": self.strength,
            "manual": self.stabilizer.locked,
            "toast": self._current_toast(),
            "scan": scan,
        }

        self.frame_ready.emit(image, state)

    # -----------------------------------------------------
    # BACKGROUND CAPTURE
    # -----------------------------------------------------

    def capture_background(self):

        if not self.running:
            return False

        start_time = time.time()

        while (
            self.running
            and time.time() - start_time < COUNTDOWN_SEC
        ):

            frame = self.read_frame()

            if frame is None:
                self.msleep(20)
                continue

            elapsed = time.time() - start_time

            remaining = max(
                0,
                int(COUNTDOWN_SEC - elapsed + 0.99),
            )

            self.emit_frame(
                frame,
                scan=(
                    remaining,
                    elapsed / max(COUNTDOWN_SEC, 0.001),
                    "BACKGROUND SCAN // STEP OUT OF FRAME",
                ),
            )

            self.msleep(10)

        if not self.running:
            return False

        frames = []
        attempts = 0

        while (
            self.running
            and len(frames) < BG_SAMPLES
            and attempts < BG_SAMPLES * 5
        ):

            attempts += 1

            frame = self.read_frame()

            if frame is not None:

                frames.append(frame)

                self.emit_frame(
                    frame,
                    scan=(
                        0,
                        1.0,
                        "CAPTURING // HOLD STILL",
                    ),
                )

            self.msleep(10)

        if not self.running or not frames:

            self.status_changed.emit(
                "ERROR // Background capture failed."
            )

            return False

        background = np.median(
            np.stack(frames),
            axis=0,
        ).astype(np.uint8)

        self.engine.set_background(background)

        self.strength = 0.0

        self.post_toast("BACKGROUND LOCKED", 2.2)

        return True

    # -----------------------------------------------------
    # CONTROLS
    # -----------------------------------------------------

    def request_mode(self, mode):

        with self.command_lock:
            self.manual_mode = mode

    def request_recapture(self):

        if self.running:
            self.recapture_requested.set()

    def request_screenshot(self):

        if self.running:
            self.screenshot_requested.set()

    def toggle_hud(self):

        self.show_hud = not self.show_hud

    def toggle_landmarks(self):

        self.show_landmarks = not self.show_landmarks

        self.post_toast(
            "LANDMARKS " + (
                "ON" if self.show_landmarks else "OFF"
            ),
            1.2,
        )

    def _apply_manual_command(self):

        with self.command_lock:

            requested = self.manual_mode
            self.manual_mode = None

        if requested is not None:

            self.stabilizer.force_mode(requested)

            self.post_toast(
                f"MANUAL // {requested}",
                1.8,
            )

    def stop(self):

        self.running = False

        if self.isRunning():
            self.wait(5000)

    def _save_screenshot(self, image):

        os.makedirs("screenshots", exist_ok=True)

        path = os.path.join(
            "screenshots",
            time.strftime("deleteme_%Y%m%d_%H%M%S.png"),
        )

        success = cv2.imwrite(path, image)

        self.post_toast(
            "SCREENSHOT SAVED" if success
            else "SCREENSHOT FAILED",
            2.0,
        )

    # -----------------------------------------------------
    # MAIN CAMERA LOOP
    # -----------------------------------------------------

    def run(self):

        self.running = True

        try:

            self.camera = self.open_camera()

            if not self.capture_background():
                return

            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            mp_selfie = mp.solutions.selfie_segmentation

            landmark_style = mp_draw.DrawingSpec(
                color=WHITE,
                thickness=2,
                circle_radius=3,
            )

            connection_style = mp_draw.DrawingSpec(
                color=CYAN,
                thickness=2,
            )

            previous_time = time.time()
            fps = 0.0
            bad_frames = 0

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as hands:

                with mp_selfie.SelfieSegmentation(
                    model_selection=1
                ) as selfie:

                    while self.running:

                        if self.recapture_requested.is_set():

                            self.recapture_requested.clear()

                            if (
                                not self.capture_background()
                                and not self.running
                            ):
                                break

                            previous_time = time.time()

                            continue

                        frame = self.read_frame()

                        if frame is None:

                            bad_frames += 1

                            if bad_frames > 120:

                                raise RuntimeError(
                                    "Camera frames aa hi nahi rahe."
                                )

                            self.msleep(20)

                            continue

                        bad_frames = 0

                        small = cv2.resize(
                            frame,
                            (PROC_W, PROC_H),
                            interpolation=cv2.INTER_AREA,
                        )

                        rgb = cv2.cvtColor(
                            small,
                            cv2.COLOR_BGR2RGB,
                        )

                        rgb.flags.writeable = False

                        # Hand gesture detection
                        hand_results = hands.process(rgb)

                        gesture = "NO HAND"

                        if hand_results.multi_hand_landmarks:

                            gesture = detect_gesture(
                                hand_results.multi_hand_landmarks[0]
                            )

                        self._apply_manual_command()

                        mode = self.stabilizer.update(gesture)

                        # Effect strength transition
                        target = TARGET_STRENGTH[mode]

                        self.strength += (
                            target - self.strength
                        ) * EASE

                        if abs(target - self.strength) < 0.01:
                            self.strength = target

                        # Apply visual effect
                        if self.strength > 0.01:

                            raw_mask = selfie.process(
                                rgb
                            ).segmentation_mask

                            mask = self.engine.update_mask(
                                small,
                                raw_mask,
                            )

                            output = self.engine.composite(
                                frame,
                                mask,
                                mode,
                                self.strength,
                                time.time(),
                            )

                        else:

                            self.engine.reset_temporal()

                            output = frame

                        # Screenshot captures the processed frame.
                        if self.screenshot_requested.is_set():

                            self.screenshot_requested.clear()

                            self._save_screenshot(output)

                        # Draw hand landmarks over the processed frame.
                        if (
                            self.show_landmarks
                            and hand_results.multi_hand_landmarks
                        ):

                            mp_draw.draw_landmarks(
                                output,
                                hand_results.multi_hand_landmarks[0],
                                mp_hands.HAND_CONNECTIONS,
                                landmark_style,
                                connection_style,
                            )

                        now = time.time()

                        instantaneous_fps = 1.0 / max(
                            now - previous_time,
                            0.001,
                        )

                        fps = (
                            instantaneous_fps
                            if fps == 0
                            else 0.9 * fps
                            + 0.1 * instantaneous_fps
                        )

                        previous_time = now

                        self.emit_frame(
                            output,
                            mode,
                            gesture,
                            fps,
                        )

        except Exception as error:

            self.status_changed.emit(
                f"ERROR // {error}"
            )

        finally:

            self.running = False

            if self.camera is not None:

                self.camera.release()
                self.camera = None

            self.status_changed.emit("SYSTEM OFFLINE")


# =========================================================
# MAIN WINDOW
# =========================================================

class DeleteMePro(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(WINDOW_TITLE)

        self.setStyleSheet(
            "QMainWindow { background-color: black; }"
            "QLabel { background-color: black; "
            "color: #9fd8b0; font-size: 18px; }"
        )

        self.camera_label = QLabel(
            "DELETEME PRO\n\nINITIALIZING CAMERA..."
        )

        self.camera_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.camera_label.setMinimumSize(320, 180)

        self.setCentralWidget(self.camera_label)

        # ==========================================
        # FLOATING CYBERPUNK RECORDING CONTROLS
        # ==========================================

        self.record_button = QPushButton(
            "●  START RECORDING",
            self.camera_label,
        )

        self.quality_button = QPushButton(
            "QUALITY · 720p ▾",
            self.camera_label,
        )

        self.screenshot_button = QPushButton(
            "▣  SCREENSHOT",
            self.camera_label,
        )

        # Recording button
        self.record_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(110, 25, 30, 225);
                color: #ff7777;
                border: 1px solid #e84b55;
                border-radius: 7px;
                padding: 10px 14px;
                font-family: Segoe UI;
                font-size: 12px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: rgba(160, 35, 45, 245);
                color: white;
            }
        """)

        # Quality selector
        self.quality_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.quality_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(20, 30, 38, 225);
                color: #9fe8f5;
                border: 1px solid #477987;
                border-radius: 7px;
                padding: 10px 12px;
                font-family: Segoe UI;
                font-size: 11px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: rgba(35, 65, 75, 245);
                border-color: #9fe8f5;
            }
        """)

        # Screenshot button
        self.screenshot_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.screenshot_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(20, 30, 25, 225);
                color: #a8e8b0;
                border: 1px solid #548b60;
                border-radius: 7px;
                padding: 10px 12px;
                font-family: Segoe UI;
                font-size: 11px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: rgba(35, 75, 45, 245);
                border-color: #a8e8b0;
            }
        """)

        # Connect buttons to existing methods
        self.record_button.clicked.connect(
            self._toggle_recording
        )

        self.quality_button.clicked.connect(
            self._cycle_record_quality
        )

        self.screenshot_button.clicked.connect(
            self._take_screenshot
        )

        # Initial positions
        self._position_controls()

        self.worker = None

        self.last_image = None
        self.last_hud = None
        self.error_message = None

        # Recording settings
        self.recording = False
        self.record_quality = "720p"
        self.record_started_at = None

        self.video_writer = None
        self.recording_path = None

        self.output_fps = 24.0

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    def start_camera(self):

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        self.last_image = None
        self.last_hud = None
        self.error_message = None

        self.camera_label.setPixmap(QPixmap())

        self.camera_label.setText(
            "DELETEME PRO\n\nINITIALIZING CAMERA..."
        )

        self.worker = CameraWorker()

        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_changed.connect(self.update_status)
        self.worker.finished.connect(self.camera_finished)

        self.worker.start()

    def update_frame(self, image, hud):

        self.last_image = image
        self.last_hud = hud

        self.display_image(image, hud)

        if self.recording:
            self._record_frame(image, hud)

    def _hud_state(self, hud):

        state = dict(hud or {})

        state["recording"] = self.recording
        state["record_quality"] = self.record_quality

        if (
            self.recording
            and self.record_started_at is not None
        ):

            state["record_seconds"] = (
                time.time() - self.record_started_at
            )

        else:

            state["record_seconds"] = 0

        return state

    def display_image(self, image, hud):

        dpr = self.devicePixelRatioF()

        size = self.camera_label.size()

        target = QSize(
            max(2, int(size.width() * dpr)),
            max(2, int(size.height() * dpr)),
        )

        pixmap = QPixmap.fromImage(image).scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        pixmap.setDevicePixelRatio(dpr)

        painter = QPainter(pixmap)

        paint_hud(
            painter,
            pixmap.width() / dpr,
            pixmap.height() / dpr,
            self._hud_state(hud),
        )

        painter.end()

        self.camera_label.setPixmap(pixmap)

    # ==========================================
    # FLOATING CONTROL ACTIONS
    # ==========================================

    def _position_controls(self):
        """Position controls at the bottom-center of the camera."""

        if not hasattr(self, "record_button"):
            return

        gap = 10
        record_w = 190
        quality_w = 155
        screenshot_w = 150
        button_h = 42

        total_w = (
            record_w
            + quality_w
            + screenshot_w
            + gap * 2
        )

        x = max(
            10,
            (self.camera_label.width() - total_w) // 2,
        )

        y = max(
            10,
            self.camera_label.height() - button_h - 24,
        )

        self.record_button.setGeometry(
            x, y, record_w, button_h
        )

        self.quality_button.setGeometry(
            x + record_w + gap,
            y,
            quality_w,
            button_h,
        )

        self.screenshot_button.setGeometry(
            x + record_w + gap + quality_w + gap,
            y,
            screenshot_w,
            button_h,
        )

    def _cycle_record_quality(self):
        """Switch between 720p and 1080p."""

        if self.record_quality == "720p":
            quality = "1080p"
        else:
            quality = "720p"

        self._set_record_quality(quality)

        self.quality_button.setText(
            f"QUALITY · {self.record_quality} ▾"
        )

    def _take_screenshot(self):
        """Save a processed screenshot."""

        if self._worker_running():
            self.worker.request_screenshot()
        else:
            self.record_button.setToolTip(
                "Start the camera before taking a screenshot."
            )

    # -----------------------------------------------------
    # VIDEO RECORDING
    # -----------------------------------------------------

    def _start_recording(self):

        if not self._worker_running():
            return

        if self.recording:
            return

        self.recording = True

        self.record_button.setText(
            "■  STOP & SAVE"
        )

        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(15, 90, 55, 235);
                color: #a8ffc5;
                border: 1px solid #35d47b;
                border-radius: 7px;
                padding: 10px 14px;
                font-family: Segoe UI;
                font-size: 12px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: rgba(20, 130, 75, 245);
                color: white;
            }
        """)

        self.record_started_at = time.time()

        self.video_writer = None
        self.recording_path = None

        if self.worker:
            self.worker.post_toast(
                f"REC STARTED // {self.record_quality}",
                2.0,
            )

    def _record_frame(self, image, hud):

        try:

            if self.record_quality == "1080p":

                width, height = 1920, 1080

            else:

                width, height = 1280, 720

            # Initialize video writer on the first recorded frame.
            if self.video_writer is None:

                os.makedirs("recordings", exist_ok=True)

                filename = time.strftime(
                    "VEIL_%Y%m%d_%H%M%S"
                )

                filename += f"_{self.record_quality}.mp4"

                self.recording_path = os.path.abspath(
                    os.path.join("recordings", filename)
                )

                fps = float(
                    (hud or {}).get("fps", 24.0) or 24.0
                )

                self.output_fps = max(
                    12.0,
                    min(30.0, fps),
                )

                writer = cv2.VideoWriter(
                    self.recording_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.output_fps,
                    (width, height),
                )

                if not writer.isOpened():

                    writer.release()

                    self.recording = False
                    self.recording_path = None

                    if self.worker:

                        self.worker.post_toast(
                            "RECORDING FAILED // CODEC ERROR",
                            3.0,
                        )

                    return

                self.video_writer = writer

            # Create the final recording frame.
            rendered = QImage(
                width,
                height,
                QImage.Format.Format_RGB888,
            )

            rendered.fill(QColor(0, 0, 0))

            painter = QPainter(rendered)

            painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform,
                True,
            )

            painter.drawImage(
                QRectF(0, 0, width, height),
                image,
            )

            # Include HUD and recording timer in the saved video.
            paint_hud(
                painter,
                width,
                height,
                self._hud_state(hud),
            )

            painter.end()

            rendered = rendered.convertToFormat(
                QImage.Format.Format_RGB888
            )

            pixels = np.frombuffer(
                rendered.bits(),
                dtype=np.uint8,
                count=width * height * 3,
            ).reshape((height, width, 3))

            bgr_frame = cv2.cvtColor(
                pixels,
                cv2.COLOR_RGB2BGR,
            )

            self.video_writer.write(bgr_frame)

        except Exception as error:

            self._stop_recording()

            if self.worker:

                self.worker.post_toast(
                    f"RECORDING ERROR // {error}",
                    4.0,
                )

    def _stop_recording(self):

        was_recording = self.recording

        self.recording = False

        self.record_button.setText(
            "●  START RECORDING"
        )

        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(110, 25, 30, 225);
                color: #ff7777;
                border: 1px solid #e84b55;
                border-radius: 7px;
                padding: 10px 14px;
                font-family: Segoe UI;
                font-size: 12px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: rgba(160, 35, 45, 245);
                color: white;
            }
        """)

        if self.video_writer is not None:

            self.video_writer.release()
            self.video_writer = None

        path = self.recording_path

        self.recording_path = None
        self.record_started_at = None

        if was_recording and self.worker:

            if path and os.path.isfile(path):

                self.worker.post_toast(
                    f"VIDEO SAVED // {os.path.basename(path)}",
                    3.0,
                )

            else:

                self.worker.post_toast(
                    "RECORDING STOPPED",
                    2.0,
                )

    def _toggle_recording(self):

        if self.recording:

            self._stop_recording()

        else:

            self._start_recording()

    def _set_record_quality(self, quality):

        if self.recording:

            if self.worker:

                self.worker.post_toast(
                    "STOP RECORDING TO CHANGE QUALITY",
                    2.0,
                )

            return

        self.record_quality = quality

        if self.worker:

            self.worker.post_toast(
                f"VIDEO QUALITY // {quality}",
                2.0,
            )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    def update_status(self, message):

        if message.startswith("ERROR"):

            self.error_message = message

    def camera_finished(self):

        if self.recording or self.video_writer is not None:
            self._stop_recording()

        self.last_image = None
        self.last_hud = None

        self.camera_label.setPixmap(QPixmap())

        text = self.error_message or "SYSTEM OFFLINE"

        self.camera_label.setText(
            f"{text}\n\nPress R to restart camera."
        )

    # -----------------------------------------------------
    # KEYBOARD CONTROLS
    # -----------------------------------------------------

    def _worker_running(self):

        return (
            self.worker is not None
            and self.worker.isRunning()
        )

    def keyPressEvent(self, event):

        key = event.key()
        worker = self.worker

        if key == Qt.Key.Key_F11:

            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

            return

        elif key == Qt.Key.Key_Escape:

            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()

            return

        elif key == Qt.Key.Key_Q:

            self.close()
            return

        elif key == Qt.Key.Key_R:

            if not self._worker_running():
                self.start_camera()

            return

        elif key == Qt.Key.Key_V:

            if self._worker_running():
                self._toggle_recording()

            return

        elif key == Qt.Key.Key_7:

            self._set_record_quality("720p")
            return

        elif key == Qt.Key.Key_8:

            self._set_record_quality("1080p")
            return

        if self._worker_running():

            if key == Qt.Key.Key_B:

                worker.request_recapture()

            elif key == Qt.Key.Key_1:

                worker.request_mode("NORMAL")

            elif key == Qt.Key.Key_2:

                worker.request_mode("INVISIBLE")

            elif key == Qt.Key.Key_3:

                worker.request_mode("GHOST")

            elif key == Qt.Key.Key_H:

                worker.toggle_hud()

            elif key == Qt.Key.Key_L:

                worker.toggle_landmarks()

            elif key == Qt.Key.Key_S:

                worker.request_screenshot()

            else:

                super().keyPressEvent(event)

        else:

            super().keyPressEvent(event)

    # -----------------------------------------------------
    # RESIZE / CLOSE
    # -----------------------------------------------------

    def resizeEvent(self, event):

        super().resizeEvent(event)

        self._position_controls()

        if self.last_image is not None:

            self.display_image(
                self.last_image,
                self.last_hud,
            )

    def closeEvent(self, event):

        if self.recording or self.video_writer is not None:
            self._stop_recording()

        if self._worker_running():
            self.worker.stop()

        event.accept()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = DeleteMePro()

    window.showFullScreen()
    window.start_camera()

    sys.exit(app.exec())