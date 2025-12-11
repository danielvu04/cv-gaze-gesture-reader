from PyQt5.QtCore import Qt, QRect, QTimer, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import QWidget, QApplication


class OverlayWidget(QWidget):
    def __init__(self, screen_width, screen_height):
        super().__init__()

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()

        self.regions = []
        self.active_index = -1
        self.gaze_point = None

        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self.update)
        self._update_timer.start(30)

        self.setMouseTracking(True)

        # Enable keyboard focus for exiting with ESC
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()


    def set_regions(self, region_data):
        self.regions = []
        for bbox, summary in region_data:
            x1, y1, x2, y2 = bbox
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            self.regions.append({"rect": rect, "summary": summary})
        self.update()

    def set_gaze(self, gaze_tuple):
        if gaze_tuple is None:
            self.gaze_point = None
        else:
            x, y = gaze_tuple
            self.gaze_point = QPoint(x, y)
        self.update()

    def set_active_region(self, idx):
        self.active_index = idx
        self.update()

    def set_summary(self, region_index, summary):
        if 0 <= region_index < len(self.regions):
            self.regions[region_index]["summary"] = summary
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        font = QFont("Sans", 11)
        painter.setFont(font)

        # Regions + labels + summary bubbles
        for i, r in enumerate(self.regions):
            rect = r["rect"]
            if i == self.active_index:
                color = QColor(255, 255, 0, 120)
                pen_color = QColor(255, 255, 0)
                pen_width = 3
            else:
                color = QColor(0, 255, 0, 60)
                pen_color = QColor(0, 255, 0)
                pen_width = 2

            painter.fillRect(rect, color)
            painter.setPen(QPen(pen_color, pen_width))
            painter.drawRect(rect)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect.adjusted(5, 5, -5, -5),
                             Qt.AlignLeft | Qt.AlignTop,
                             f"Region {i+1}")

            summary = r.get("summary", "")
            if summary:
                bubble_width = int(self.screen_width * 0.3)
                bubble_height = 160
                bubble_x = rect.right() + 10
                bubble_y = rect.top()
                if bubble_x + bubble_width > self.screen_width:
                    bubble_x = rect.left() - 10 - bubble_width
                bubble_rect = QRect(bubble_x, bubble_y, bubble_width, bubble_height)

                painter.setBrush(QColor(0, 0, 0, 200))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawRoundedRect(bubble_rect, 10, 10)

                painter.setPen(QColor(255, 255, 255))
                summary_font = QFont("Sans", 9)
                painter.setFont(summary_font)

                text = summary
                max_chars = 50
                lines = []
                while len(text) > max_chars:
                    split_at = text.rfind(" ", 0, max_chars)
                    if split_at == -1:
                        split_at = max_chars
                    lines.append(text[:split_at])
                    text = text[split_at + 1:]
                lines.append(text)

                text_y = bubble_rect.top() + 20
                for line in lines[:6]:
                    painter.drawText(bubble_rect.left() + 10, text_y, line)
                    text_y += 18

        # Gaze indicator
        if self.gaze_point is not None:
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setBrush(QColor(255, 0, 0, 180))
            painter.drawEllipse(self.gaze_point, 8, 8)
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.gaze_point + QPoint(10, 0), "Gaze")
    
    def keyPressEvent(self, event):
        # Exit on ESC key
        if event.key() == Qt.Key_Escape:
            QApplication.instance().quit()
