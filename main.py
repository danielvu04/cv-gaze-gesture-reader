import sys
import mss

#handle pytesseract path
import pytesseract
#replace this path with your own tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from PyQt5.QtWidgets import QApplication

from modules.overlay import OverlayWidget
from modules.readingpipeline import ReadingPipeline

def get_primary_monitor_index():
    with mss.mss() as sct:
        monitors = sct.monitors
        for i, mon in enumerate(monitors):
            if mon['left'] == 0 and mon['top'] == 0:
                return i  
    return 1 

def main():
    app = QApplication(sys.argv)

    screen = app.primaryScreen()
    geometry = screen.geometry()
    sw = geometry.width()
    sh = geometry.height()

    overlay = OverlayWidget(sw, sh)

    pipeline = ReadingPipeline(monitor_index = get_primary_monitor_index())
    pipeline.regionsDefined.connect(overlay.set_regions)
    pipeline.gazeUpdated.connect(overlay.set_gaze)
    pipeline.activeRegionChanged.connect(overlay.set_active_region)
    pipeline.summaryUpdated.connect(overlay.set_summary)

    pipeline.start()

    exit_code = app.exec_()
    pipeline.stop()
    pipeline.wait()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
