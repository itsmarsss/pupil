"""Transparent, click-through window over everything: gaze dot, snapped target, status line."""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class Overlay(QWidget):
    def __init__(self, geometry):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(geometry)
        self.gaze = None
        self.draw_gaze = True  # off when the tracker draws its own ring
        self.target = None
        self.candidates = []
        self.status = ""
        self.heard = ""
        self.mic = "off"
        self.mic_device = ""
        self.gate = ""  # lip gate state, when it's on
        self.warning = ""
        self.flash = None  # (x, y) of the last click, drawn briefly
        self.panel = None  # listen-only demo: dict(gate_open, reason, rec, voice, level, misses), drawn at the top

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        for c in self.candidates:
            if c != self.target:
                p.setPen(QPen(QColor(0, 200, 255, 70), 1))
                p.drawRoundedRect(QRectF(c.x, c.y, c.w, c.h), 4, 4)
        if self.target:
            p.setPen(QPen(QColor(0, 200, 255, 220), 3))
            p.drawRoundedRect(QRectF(self.target.x - 2, self.target.y - 2, self.target.w + 4, self.target.h + 4), 6, 6)
        if self.gaze and self.draw_gaze:
            p.setPen(QPen(QColor(255, 60, 60, 160), 3))
            p.drawEllipse(QPointF(*self.gaze), 14, 14)
        if self.flash:
            p.setPen(QPen(QColor(80, 255, 120, 230), 4))
            p.drawEllipse(QPointF(*self.flash), 22, 22)

        if self.panel:
            self._draw_panel(p)

        mic = f"mic: {self.mic_device} · {self.mic}" if self.mic_device else f"mic: {self.mic}"
        if self.gate:
            mic += f" · {self.gate}"
        lines = [s for s in (self.warning, mic, f'heard: "{self.heard}"' if self.heard else "",
                             self.status) if s]
        p.setFont(QFont("Helvetica Neue", 14))
        text = "   |   ".join(lines)
        width = p.fontMetrics().horizontalAdvance(text) + 32
        box = QRectF((self.width() - width) / 2, self.height() - 64, width, 36)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 24, 200))
        p.drawRoundedRect(box, 18, 18)
        p.setPen(QColor(240, 240, 240))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_panel(self, p):
        """Gate open/closed, whether audio is going to the recognizer, the mic level, and possible misses."""
        d = self.panel
        box = QRectF((self.width() - 620) / 2, 44, 620, 92)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 24, 220))
        p.drawRoundedRect(box, 16, 16)

        # gate pill
        pill = QRectF(box.x() + 14, box.y() + 14, 210, 38)
        p.setBrush(QColor(40, 170, 90) if d["gate_open"] else QColor(90, 90, 98))
        p.drawRoundedRect(pill, 19, 19)
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Helvetica Neue", 17, QFont.Weight.Bold))
        p.drawText(pill, Qt.AlignmentFlag.AlignCenter, "GATE OPEN" if d["gate_open"] else "GATE CLOSED")
        p.setFont(QFont("Helvetica Neue", 12))
        p.setPen(QColor(200, 200, 205))
        p.drawText(QRectF(pill.x(), pill.bottom() + 2, pill.width(), 20), Qt.AlignmentFlag.AlignCenter, d["reason"])

        # recording dot: audio is actually being sent
        x = pill.right() + 24
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(235, 50, 50) if d["rec"] else QColor(70, 70, 76))
        p.drawEllipse(QPointF(x + 10, pill.center().y()), 10, 10)
        p.setPen(QColor(240, 240, 240) if d["rec"] else QColor(130, 130, 136))
        p.setFont(QFont("Helvetica Neue", 15, QFont.Weight.Bold))
        p.drawText(QRectF(x + 26, pill.y(), 70, pill.height()), Qt.AlignmentFlag.AlignVCenter, "REC")

        # mic level: is there speech at the mic at all
        x += 110
        meter = QRectF(x, pill.center().y() - 7, 150, 14)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(60, 60, 66))
        p.drawRoundedRect(meter, 7, 7)
        p.setBrush(QColor(80, 200, 255) if d["voice"] else QColor(120, 120, 128))
        p.drawRoundedRect(QRectF(meter.x(), meter.y(), meter.width() * min(1.0, d["level"]), meter.height()), 7, 7)
        p.setPen(QColor(200, 200, 205))
        p.setFont(QFont("Helvetica Neue", 12))
        p.drawText(QRectF(meter.x(), meter.bottom() + 6, meter.width(), 20), Qt.AlignmentFlag.AlignCenter,
                   "voice" if d["voice"] else "mic")

        # possible misses: voice at the mic while nothing was being sent
        x = meter.right() + 16
        warn = d["voice"] and not d["rec"]
        p.setPen(QColor(255, 160, 40) if warn or d["misses"] else QColor(130, 130, 136))
        p.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold if warn else QFont.Weight.Normal))
        p.drawText(QRectF(x, box.y() + 10, box.right() - x - 10, box.height() - 20), Qt.AlignmentFlag.AlignCenter,
                   f"{'MISSING?' if warn else 'possible'}\nmisses: {d['misses']}")
