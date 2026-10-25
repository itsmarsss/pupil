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
