import cv2
import mediapipe as mp

class GazeTracker:
    """
    Gaze tracker using MediaPipe Face Mesh + iris landmarks.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Correct indices
        self.right_iris_indices = [469, 470, 471, 472]
        self.right_eye_outer = 33
        self.right_eye_inner = 133
        self.right_eye_upper = 159
        self.right_eye_lower = 145

        self.last_iris_px = None

    def process(self, frame):
        """
        Returns:
            gaze_point (gx, gy) in normalized eye-local coords (0..1), or None if no face.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        face = result.multi_face_landmarks[0].landmark

        # Iris center (right eye)
        iris_x = sum(face[i].x for i in self.right_iris_indices) / len(self.right_iris_indices)
        iris_y = sum(face[i].y for i in self.right_iris_indices) / len(self.right_iris_indices)

        # Right eye box
        outer = face[self.right_eye_outer]
        inner = face[self.right_eye_inner]
        upper = face[self.right_eye_upper]
        lower = face[self.right_eye_lower]

        eye_min_x = min(outer.x, inner.x)
        eye_max_x = max(outer.x, inner.x)
        eye_min_y = min(upper.y, lower.y)
        eye_max_y = max(upper.y, lower.y)

        eye_width = eye_max_x - eye_min_x
        eye_height = eye_max_y - eye_min_y
        if eye_width < 1e-4 or eye_height < 1e-4:
            return None

        gx = (iris_x - eye_min_x) / eye_width
        gy = (iris_y - eye_min_y) / eye_height

        # Safety clamp
        gx = max(0.0, min(1.0, gx))
        gy = max(0.0, min(1.0, gy))

        px = int(iris_x * w)
        py = int(iris_y * h)
        self.last_iris_px = (px, py)

        return (gx, gy)

    def draw_debug(self, frame, gaze_point):
        if self.last_iris_px is not None:
            cv2.circle(frame, self.last_iris_px, 6, (0, 255, 255), -1)

    def release(self):
        self.mesh.close()
