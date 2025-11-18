import cv2
import mediapipe as mp


class GazeTracker:
    """
    Gaze tracker using MediaPipe Face Mesh + iris landmarks.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True to get iris landmarks
        self.mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Iris indices for right eye (MediaPipe standard indices)
        self.right_iris_indices = [474, 475, 476, 477]

    def process(self, frame):
        """
        Returns:
            gaze_point (x, y) in image coordinates, or None if no face.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        face_landmarks = result.multi_face_landmarks[0]

        iris_x = 0.0
        iris_y = 0.0
        for idx in self.right_iris_indices:
            lm = face_landmarks.landmark[idx]
            iris_x += lm.x
            iris_y += lm.y

        iris_x /= len(self.right_iris_indices)
        iris_y /= len(self.right_iris_indices)

        # map normalized [0, 1] to pixel coordinates
        px = int(iris_x * w)
        py = int(iris_y * h)
        return (px, py)

    def draw_debug(self, frame, gaze_point):
        if gaze_point is not None:
            cv2.circle(frame, gaze_point, 6, (0, 255, 255), -1)

    def release(self):
        self.mesh.close()
