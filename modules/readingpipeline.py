import time
import numpy as np
import pyautogui
import cv2

from PyQt5.QtCore import QThread, pyqtSignal, QCoreApplication
from modules.gaze import GazeTracker
from modules.gestures import GestureRecognizer
from modules.ocr import extract_text_from_region, summarize_text
from modules.fusion import FusionEngine
from modules.metrics import MetricsTracker
from modules.layout import detect_text_regions, create_fallback_regions  # Updated: Include fallback
from modules.calibration import run_calibration, apply_affine
from modules.screencap import ScreenCapture


class ReadingPipeline(QThread):
    # Signals to update the UI
    regionsDefined = pyqtSignal(object)    # list of (bbox, summary)
    gazeUpdated = pyqtSignal(object)       # (x, y) in screen coords, or None
    activeRegionChanged = pyqtSignal(int)  # index, or -1
    summaryUpdated = pyqtSignal(int, str)  # region_index, summary text

    def __init__(self, monitor_index=1, parent=None):
        super().__init__(parent)
        self.monitor_index = monitor_index
        self.running = True

        self.smooth_gaze_screen = None  # Initial smoothed point
        self.alpha = 0.7  
        self.metrics = MetricsTracker()

        # Track how long open palm has been held for exit gesture
        self.open_palm_frames = 0

    def run(self):
        # Webcam for gaze + gestures
        cam_cap = cv2.VideoCapture(0)
        if not cam_cap.isOpened():
            print("Could not open webcam")
            return

        # Discover full screen size
        temp_screen = ScreenCapture(self.monitor_index)
        full_frame = temp_screen.grab()
        full_h, full_w, _ = full_frame.shape
        temp_screen.release()

        # Capture from chosen monitor
        screen = ScreenCapture(self.monitor_index)
        screen_frame = screen.grab()
        sh, sw, _ = screen_frame.shape

        # Define adaptive reading regions + fusion (based on initial OCR)
        regions = detect_text_regions(screen_frame)  # New: Dynamic detection
        if not regions:  # Fallback if no text
            regions = create_fallback_regions(sw, sh)
        fusion = FusionEngine(regions)

        # Initialize trackers
        gaze_tracker = GazeTracker()
        gesture_recognizer = GestureRecognizer()

        # Send initial regions to UI overlay
        simple_regions = [(r.bbox, r.summary) for r in regions]
        self.regionsDefined.emit(simple_regions)

        # Calibration
        M = run_calibration(gaze_tracker, cam_cap, sw, sh, self.metrics)
        if M is None:
            print("Calibration failed or aborted.")
            cam_cap.release()
            gaze_tracker.release()
            gesture_recognizer.release()
            screen.release()
            return

        print("Calibration done. Starting main loop.")

        while self.running:
            ret_cam, cam_frame = cam_cap.read()
            if not ret_cam:
                break

            screen_frame = screen.grab()

            # Gaze
            gaze_cam = gaze_tracker.process(cam_frame)
            gaze_screen = apply_affine(M, gaze_cam)
            if gaze_screen is not None:
                if self.smooth_gaze_screen is None:
                    self.smooth_gaze_screen = gaze_screen 
                else:
                    prev_x, prev_y = self.smooth_gaze_screen
                    curr_x, curr_y = gaze_screen
                    smooth_x = self.alpha * curr_x + (1 - self.alpha) * prev_x
                    smooth_y = self.alpha * curr_y + (1 - self.alpha) * prev_y
                    self.smooth_gaze_screen = (int(smooth_x), int(smooth_y))
                fusion.update_gaze(self.smooth_gaze_screen)  
                self.gazeUpdated.emit(self.smooth_gaze_screen)  
            else:
                self.smooth_gaze_screen = None
                fusion.update_gaze(None)
                self.gazeUpdated.emit(None)

            active_index = fusion.active_index if fusion.active_index is not None else -1
            self.activeRegionChanged.emit(active_index)

            # Gestures
            gesture_info = gesture_recognizer.process(cam_frame)

            # Exit: open palm must be held for N frames
            if gesture_info["open_palm"]:
                self.open_palm_frames += 1
            else:
                self.open_palm_frames = 0

            EXIT_HOLD_FRAMES = 60
            if self.open_palm_frames >= EXIT_HOLD_FRAMES:
                print("Open palm held - exiting.")
                self.running = False
                QCoreApplication.quit()
                break

            # Scroll: pinch + swipe (recompute regions after scroll)
            scroll_triggered = False
            if gesture_info["swipe_up_trigger"] and gesture_info["is_pinch"]:
                pyautogui.scroll(-500)          # negative = scroll down
                self.metrics.log_swipe("down")
                scroll_triggered = True
            if gesture_info["swipe_down_trigger"] and gesture_info["is_pinch"]:
                pyautogui.scroll(500)           # positive = scroll up
                self.metrics.log_swipe("up")
                scroll_triggered = True
            
            if scroll_triggered:
                time.sleep(0.5)  # Wait for scroll 
                new_frame = screen.grab()
                new_regions = detect_text_regions(new_frame)
                if new_regions:
                    regions[:] = new_regions
                    fusion.regions = regions
                    simple_regions = [(r.bbox, r.summary) for r in regions]
                    self.regionsDefined.emit(simple_regions) 

            # Summarize: thumbs up
            region_index = fusion.should_trigger_summary(
                gesture_info["thumbs_up_trigger"]
            )
            if region_index is not None:
                region = regions[region_index]
                if not region.text: 
                    t0 = time.time()
                    text = extract_text_from_region(screen_frame, region.bbox)
                    region.text = text
                    region.summary = summarize_text(text)
                    t1 = time.time()
                    latency_ms = (t1 - t0) * 1000.0
                    self.metrics.log_summary_latency(latency_ms)
                    print(f"[Region {region_index + 1}] OCR text:\n{text}\n")
                    print(f"[Region {region_index + 1}] Summary:\n{region.summary}\n")
                    print(f"Summary latency: {latency_ms:.1f} ms")
                else:
                    print(f"[Region {region_index + 1}] Using cached summary:\n{region.summary}\n")
                
                self.summaryUpdated.emit(region_index, region.summary)

            # Clear summary: thumbs down clears current active region
            if gesture_info["thumbs_down_trigger"] and fusion.active_index is not None:
                idx = fusion.active_index
                regions[idx].summary = ""
                regions[idx].text = ""  
                self.summaryUpdated.emit(idx, "")

            # Webcam debug window
            gaze_tracker.draw_debug(cam_frame, gaze_cam)
            gesture_recognizer.draw_debug(cam_frame, gesture_info)
            cv2.imshow("Webcam Debug", cam_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        # Cleanup
        cam_cap.release()
        gaze_tracker.release()
        gesture_recognizer.release()
        screen.release()
        cv2.destroyAllWindows()

        print(self.metrics.report())

    def stop(self):
        self.running = False