import cv2
import math
import mediapipe as mp
from collections import deque

MIN_SWIPE_DIST = 0.12
PINCH_THRESH = 0.05

# Tunable thresholds
OPEN_SPREAD_THRESH = 0.14    # min horizontal spread between index and pinky
THUMB_Y_MARGIN = 0.03        # how far thumb tip must be above/below MCP
PIP_SEP_TOL = 0.02           # finger straightness for open palm
CURL_TOL = 0.015             # finger curl tolerance for thumbs up/down


class GestureRecognizer:
    """
    Hand gesture recognizer.

    Outputs:
      - pinch: edge-triggered boolean when pinch starts
      - is_pinch: current pinch state
      - swipe_up_trigger / swipe_down_trigger: edge-triggered vertical swipes
      - open_palm: stable open-palm state
      - thumbs_up_trigger / thumbs_down_trigger: edge-triggered on stable change
    """

    def __init__(
        self,
        pinch_threshold: float = PINCH_THRESH,
        history_len: int = 7,
        swipe_threshold: float = MIN_SWIPE_DIST,
        open_stable_frames: int = 5,
        thumb_stable_frames: int = 3,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.pinch_threshold = pinch_threshold
        self.prev_pinch = False

        # Swipe detection
        self.history_len = history_len
        self.swipe_threshold = swipe_threshold
        self.center_history = deque(maxlen=history_len)
        self.cooldown = 0

        # Stable gesture detection (temporal smoothing)
        self.open_hist = deque(maxlen=open_stable_frames)
        self.thumbsup_hist = deque(maxlen=thumb_stable_frames)
        self.thumbsdown_hist = deque(maxlen=thumb_stable_frames)
        self.open_stable_frames = open_stable_frames
        self.thumb_stable_frames = thumb_stable_frames

        # Previous stable states for edge-trigger
        self.prev_open_stable = False
        self.prev_thumbs_up_stable = False
        self.prev_thumbs_down_stable = False

    def _finger_extended(self, lm, tip_idx, pip_idx, mcp_idx, tol=PIP_SEP_TOL) -> bool:
        """Finger straight and above its base in image space."""
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        mcp = lm[mcp_idx]

        # Vertical distances (y increases downward)
        tip_above_pip = pip.y - tip.y
        pip_above_mcp = mcp.y - pip.y

        # Require both segments to have upward separation
        return tip_above_pip > tol and pip_above_mcp > tol * 0.5

    def _finger_curled(self, lm, tip_idx, pip_idx, tol=CURL_TOL) -> bool:
        """Finger tip at or below PIP (curled)."""
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        return tip.y >= pip.y - tol

    def _is_open_palm(self, landmarks) -> bool:
        if landmarks is None:
            return False

        lm = landmarks.landmark
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        mcps = [5, 9, 13, 17]

        extended = 0
        for t, p, m in zip(tips, pips, mcps):
            if self._finger_extended(lm, t, p, m):
                extended += 1

        # Require all 4 fingers extended
        if extended < 4:
            return False

        # Require good lateral spread
        idx_tip = lm[8]
        pinky_tip = lm[20]
        spread_x = abs(idx_tip.x - pinky_tip.x)
        return spread_x > OPEN_SPREAD_THRESH

    def _is_thumbs_up(self, landmarks) -> bool:
        if landmarks is None:
            return False

        # Thumb joints
        lm = landmarks.landmark
        thumb_tip = lm[4]
        thumb_mcp = lm[2]

        thumb_extended = thumb_tip.y < thumb_mcp.y - THUMB_Y_MARGIN

        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        curled = 0
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            if self._finger_curled(lm, tip_idx, pip_idx):
                curled += 1

        if not (thumb_extended and curled >= 3):
            return False

        # Thumb clearly above other fingers
        min_other_pip_y = min(lm[p].y for p in finger_pips)
        return thumb_tip.y < min_other_pip_y - 0.02

    def _is_thumbs_down(self, landmarks) -> bool:
        if landmarks is None:
            return False

        lm = landmarks.landmark
        thumb_tip = lm[4]
        thumb_mcp = lm[2]

        thumb_extended_down = thumb_tip.y > thumb_mcp.y + THUMB_Y_MARGIN

        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        curled = 0
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            if self._finger_curled(lm, tip_idx, pip_idx):
                curled += 1

        if not (thumb_extended_down and curled >= 3):
            return False

        # Thumb clearly below other fingers
        max_other_pip_y = max(lm[p].y for p in finger_pips)
        return thumb_tip.y > max_other_pip_y + 0.02

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        is_pinch = False
        landmarks = None
        swipe_up_trigger = False
        swipe_down_trigger = False

        open_palm = False
        thumbs_up = False
        thumbs_down = False

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            lm = landmarks.landmark

            # Pinch: thumb–index distance
            thumb = lm[4]
            index = lm[8]
            dx = thumb.x - index.x
            dy = thumb.y - index.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.pinch_threshold:
                is_pinch = True

            # Center for swipe
            wrist = lm[0]
            center = (wrist.x, wrist.y)
            self.center_history.append(center)

            # Raw gesture detection
            open_palm = self._is_open_palm(landmarks)
            thumbs_up = self._is_thumbs_up(landmarks)
            thumbs_down = self._is_thumbs_down(landmarks)
        else:
            # No hand: reset histories so stale detections do not linger
            self.center_history.clear()
            self.open_hist.clear()
            self.thumbsup_hist.clear()
            self.thumbsdown_hist.clear()

        # Pinch edge trigger
        pinch_trigger = is_pinch and not self.prev_pinch
        self.prev_pinch = is_pinch

        # Swipe detection
        if self.cooldown > 0:
            self.cooldown -= 1
        elif len(self.center_history) >= self.history_len:
            x0, y0 = self.center_history[0]
            x1, y1 = self.center_history[-1]
            dy = y1 - y0
            if abs(dy) > self.swipe_threshold:
                if dy < 0:
                    swipe_up_trigger = True
                else:
                    swipe_down_trigger = True
                self.cooldown = 10

        # Temporal smoothing for gestures
        if self.open_hist.maxlen is not None:
            self.open_hist.append(1 if open_palm else 0)
        if self.thumbsup_hist.maxlen is not None:
            self.thumbsup_hist.append(1 if thumbs_up else 0)
        if self.thumbsdown_hist.maxlen is not None:
            self.thumbsdown_hist.append(1 if thumbs_down else 0)

        open_palm_stable = sum(self.open_hist) >= self.open_stable_frames
        thumbs_up_stable = sum(self.thumbsup_hist) >= self.thumb_stable_frames
        thumbs_down_stable = sum(self.thumbsdown_hist) >= self.thumb_stable_frames

        # Priority: thumbs up/down override open palm
        open_palm = open_palm_stable and not (thumbs_up_stable or thumbs_down_stable)
        thumbs_up = thumbs_up_stable
        thumbs_down = thumbs_down_stable

        # Edge triggers for thumbs up/down
        thumbs_up_trigger = thumbs_up and not self.prev_thumbs_up_stable
        thumbs_down_trigger = thumbs_down and not self.prev_thumbs_down_stable

        # Update previous stable states
        self.prev_open_stable = open_palm
        self.prev_thumbs_up_stable = thumbs_up
        self.prev_thumbs_down_stable = thumbs_down

        return {
            "pinch": pinch_trigger,
            "is_pinch": is_pinch,
            "swipe_up_trigger": swipe_up_trigger,
            "swipe_down_trigger": swipe_down_trigger,
            "open_palm": open_palm,
            "thumbs_up_trigger": thumbs_up_trigger,
            "thumbs_down_trigger": thumbs_down_trigger,
            "landmarks": landmarks,
        }

    def draw_debug(self, frame, info):
        h, w, _ = frame.shape
        if info["landmarks"] is not None:
            for lm in info["landmarks"].landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        y0 = 30
        step = 25

        def line(text, color):
            nonlocal y0
            cv2.putText(
                frame,
                text,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            y0 += step

        if info["is_pinch"]:
            line("PINCH", (0, 255, 0))
        if info["swipe_up_trigger"]:
            line("SWIPE UP", (255, 255, 0))
        if info["swipe_down_trigger"]:
            line("SWIPE DOWN", (255, 255, 0))
        if info["open_palm"]:
            line("OPEN PALM (stable)", (0, 255, 255))
        if info["thumbs_up_trigger"]:
            line("THUMBS UP (trigger)", (0, 0, 255))
        if info["thumbs_down_trigger"]:
            line("THUMBS DOWN (trigger)", (0, 0, 255))

    def release(self):
        self.hands.close()
