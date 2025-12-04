import cv2
import mediapipe as mp

# MediaPipe Hands setup (same as your project)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def is_fist(landmarks, handedness_label='Right'):
    """
    Custom fist detection:
    - Fingers (index to pinky): Check if tip.y > pip.y (curled down).
    - Thumb: Check if tip is tucked (x-position relative to MCP for handedness).
    """
    if landmarks is None:
        return False

    # Finger indices
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints
    curl_threshold = 0.05  # Adjust if too strict/loose

    # Count curled fingers (tip below PIP in y, assuming standard orientation)
    curled_count = 0
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip_y = landmarks.landmark[tip_idx].y
        pip_y = landmarks.landmark[pip_idx].y
        if tip_y > pip_y + curl_threshold:  # Curled down
            curled_count += 1

    # Thumb tucked: For right hand, tip.x < IP.x; for left, tip.x > IP.x
    thumb_tip_x = landmarks.landmark[4].x
    thumb_ip_x = landmarks.landmark[3].x  # Thumb IP joint
    if handedness_label == 'Right':
        thumb_tucked = thumb_tip_x < thumb_ip_x - curl_threshold
    else:
        thumb_tucked = thumb_tip_x > thumb_ip_x + curl_threshold

    # Fist if all 4 fingers curled and thumb tucked
    return curled_count >= 4 and thumb_tucked

# Webcam loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    fist_detected = False
    handedness_label = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get handedness
            handedness_label = handedness.classification[0].label  # 'Left' or 'Right'

            # Check for fist
            if is_fist(hand_landmarks, handedness_label):
                fist_detected = True

    # Display result
    if fist_detected:
        cv2.putText(frame, 'FIST DETECTED', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Fist Detection Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()