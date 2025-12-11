import cv2
import math
import mediapipe as mp
from collections import deque


class GestureRecognizer:
    """
    Hand gesture recognizer.

    Outputs:
      - pinch: edge-triggered boolean when pinch starts
      - is_pinch: current pinch state
      - swipe_up_trigger / swipe_down_trigger: edge-triggered vertical swipes
      - open_palm: if all fingers extended
    """

    def __init__(self, pinch_threshold=0.05, history_len=7, swipe_threshold=0.12, exit_threshold=0.1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pinch_threshold = pinch_threshold
        self.prev_pinch = False
        
        # For thumbs-up / thumbs-down edge triggers
        self.prev_thumbs_up = False
        self.prev_thumbs_down = False

        # For swipe detection
        self.history_len = history_len
        self.swipe_threshold = swipe_threshold
        self.center_history = deque(maxlen=history_len)
        self.cooldown = 0
        
        self.exit_threshold = exit_threshold

    def _is_open_palm(self, landmarks):
        if landmarks is None:
            return False

        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        extended_count = 0
        for tip_idx, mcp_idx in zip(tips, mcps):
            tip = landmarks.landmark[tip_idx]
            mcp = landmarks.landmark[mcp_idx]
            if tip.y < mcp.y:  # in image coords, smaller y is higher
                extended_count += 1
        return extended_count >= 3
    
    def _is_thumbs_up(self, landmarks):
        if landmarks is None:
            return False

        # Thumb joints
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]

        # Other fingers: index/middle/ring/pinky tips & PIP joints
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        # Thumb pointing up: tip significantly higher (smaller y) than MCP
        thumb_up = thumb_tip.y < thumb_mcp.y - 0.05

        # Other fingers curled: tips lower (greater y) than PIP joints
        curled = 0
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip = landmarks.landmark[tip_idx]
            pip = landmarks.landmark[pip_idx]
            if tip.y > pip.y:
                curled += 1

        return thumb_up and curled >= 3

    def _is_thumbs_down(self, landmarks):
        if landmarks is None:
            return False

        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]

        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        # Thumb pointing down: tip significantly lower (larger y) than MCP
        thumb_down = thumb_tip.y > thumb_mcp.y + 0.05

        # Other fingers curled
        curled = 0
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip = landmarks.landmark[tip_idx]
            pip = landmarks.landmark[pip_idx]
            if tip.y > pip.y:
                curled += 1

        return thumb_down and curled >= 3


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

            # Pinch
            thumb = landmarks.landmark[4]
            index = landmarks.landmark[8]
            dx = thumb.x - index.x
            dy = thumb.y - index.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.pinch_threshold:
                is_pinch = True

            # Center for swipe
            wrist = landmarks.landmark[0]
            center = (wrist.x, wrist.y)
            self.center_history.append(center)

            # Gestures
            open_palm = self._is_open_palm(landmarks)
            thumbs_up = self._is_thumbs_up(landmarks)
            thumbs_down = self._is_thumbs_down(landmarks)

        else:
            self.center_history.clear()

        # Pinch edge-trigger
        pinch_trigger = is_pinch and not self.prev_pinch
        self.prev_pinch = is_pinch

        # Swipe detection
        if self.cooldown > 0:
            self.cooldown -= 1
        elif len(self.center_history) >= self.history_len:
            x0, y0 = self.center_history[0]
            x1, y1 = self.center_history[-1]
            dy = y1 - y0  # positive = moved down

            if abs(dy) > self.swipe_threshold:
                if dy < 0:
                    swipe_up_trigger = True
                else:
                    swipe_down_trigger = True
                self.cooldown = 10

        # Thumbs up / down edge triggers
        thumbs_up_trigger = thumbs_up and not self.prev_thumbs_up
        thumbs_down_trigger = thumbs_down and not self.prev_thumbs_down
        self.prev_thumbs_up = thumbs_up
        self.prev_thumbs_down = thumbs_down

        return {
            "pinch": pinch_trigger,
            "is_pinch": is_pinch,
            "swipe_up_trigger": swipe_up_trigger,
            "swipe_down_trigger": swipe_down_trigger,
            "open_palm": open_palm,                 # used for exit
            "thumbs_up_trigger": thumbs_up_trigger, # summarize
            "thumbs_down_trigger": thumbs_down_trigger, # clear
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
        if info["is_pinch"]:
            cv2.putText(frame, "PINCH", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y0 += step
        if info["swipe_up_trigger"]:
            cv2.putText(frame, "SWIPE UP", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y0 += step
        if info["swipe_down_trigger"]:
            cv2.putText(frame, "SWIPE DOWN", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y0 += step
        if info["open_palm"]:
            cv2.putText(frame, "OPEN PALM", (10, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (0, 255, 255), 2)
        if info.get("exit_trigger"):
            cv2.putText(frame, "EXIT GESTURE", (10, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (0, 0, 255), 2)
        if info.get("thumbs_up_trigger"):
            cv2.putText(frame, "THUMBS UP", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y0 += step
        if info.get("thumbs_down_trigger"):
            cv2.putText(frame, "THUMBS DOWN", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def release(self):
        self.hands.close()
