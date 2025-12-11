import mss
import numpy as np
import cv2


class ScreenCapture:
    """
    Simple full-screen capture using mss.
    Captures the primary monitor and returns a BGR frame.
    """

    def __init__(self, monitor_index=1):
        self.sct = mss.mss()
        monitors = self.sct.monitors
        # monitor_index=1 usually primary monitor
        if monitor_index < 0 or monitor_index >= len(monitors):
            monitor_index = 1
        self.monitor = monitors[monitor_index]

    def grab(self):
        """
        Returns a BGR frame (numpy array) of the current screen.
        """
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)  # BGRA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def release(self):
        self.sct.close()
