import sys
from PyQt5.QtWidgets import QApplication

from modules.overlay import OverlayWidget
from modules.readingpipeline import ReadingPipeline


def main():
    app = QApplication(sys.argv)

    screen = app.primaryScreen()
    geometry = screen.geometry()
    sw = geometry.width()
    sh = geometry.height()

    overlay = OverlayWidget(sw, sh)

    pipeline = ReadingPipeline(monitor_index=1)
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
