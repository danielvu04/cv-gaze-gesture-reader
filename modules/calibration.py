import numpy as np
import cv2
from typing import List, Tuple, Optional
from modules.metrics import MetricsTracker

CamPoint = Tuple[int, int]
ScreenPoint = Tuple[int, int]

def run_calibration(
    gaze_tracker,
    cam_cap,
    screen_width: int,
    screen_height: int,
    metrics: Optional[MetricsTracker] = None,
) -> Optional[np.ndarray]:
    """
    Run multi-point gaze calibration.

    Returns:
        3x3 affine matrix M mapping camera gaze points to screen coordinates,
        or None if calibration aborted/failed.
    """

    print("Starting calibration for Qt overlay")
    print("Look at each dot and press SPACE in the OpenCV windows. Press q to abort.")

    # 9-point grid
    norm_points = [
        (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),
        (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
        (0.15, 0.85), (0.50, 0.85), (0.85, 0.85),
    ]

    targets: List[ScreenPoint] = [
        (int(screen_width * nx), int(screen_height * ny))
        for (nx, ny) in norm_points
    ]

    cam_points: List[CamPoint] = []
    screen_points: List[ScreenPoint] = []

    for idx, (sx, sy) in enumerate(targets):
        samples: List[CamPoint] = []
        print(f"Calibration target {idx + 1}/{len(targets)} at ({sx}, {sy})")

        while True:
            ret_cam, cam_frame = cam_cap.read()
            if not ret_cam:
                continue

            gaze_cam = gaze_tracker.process(cam_frame)
            if gaze_cam is not None:
                samples.append(gaze_cam)
                gaze_tracker.draw_debug(cam_frame, gaze_cam)

            calib_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.circle(calib_frame, (sx, sy), 25, (0, 255, 0), -1)
            cv2.putText(
                calib_frame,
                f"Calibration {idx + 1}/{len(targets)}: Look at the dot and press SPACE",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Calibration - Screen", calib_frame)
            cv2.imshow("Calibration - Webcam", cam_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if len(samples) > 5:
                    avg_x = int(sum(p[0] for p in samples) / len(samples))
                    avg_y = int(sum(p[1] for p in samples) / len(samples))
                    cam_points.append((avg_x, avg_y))
                    screen_points.append((sx, sy))
                    print(f"Captured {len(samples)} samples. Avg cam gaze: ({avg_x}, {avg_y})")
                    break
                else:
                    print("Not enough samples, keep looking and try again.")
            elif key == ord('q'):
                print("Calibration aborted.")
                cv2.destroyWindow("Calibration - Screen")
                cv2.destroyWindow("Calibration - Webcam")
                return None

    cv2.destroyWindow("Calibration - Screen")
    cv2.destroyWindow("Calibration - Webcam")

    if len(cam_points) < 3:
        print("Calibration failed, not enough points.")
        return None

    M = solve_affine_mapping(cam_points, screen_points)
    print("Calibration matrix:\n", M)

    # Compute calibration errors in pixels
    errors_px = []
    for (cx, cy), (sx_true, sy_true) in zip(cam_points, screen_points):
        sx_pred, sy_pred = apply_affine(M, (cx, cy))
        err = np.sqrt((sx_pred - sx_true) ** 2 + (sy_pred - sy_true) ** 2)
        errors_px.append(err)

    if metrics is not None:
        metrics.log_calibration_errors(errors_px)

    if errors_px:
        avg_err = sum(errors_px) / len(errors_px)
        max_err = max(errors_px)
        print(f"Calibration error (pixels): avg = {avg_err:.1f}, max = {max_err:.1f}")

    return M


def solve_affine_mapping(cam_points: List[CamPoint], screen_points: List[ScreenPoint]) -> np.ndarray:
    cam_pts = np.array(cam_points, dtype=np.float32)
    scr_pts = np.array(screen_points, dtype=np.float32)

    n = cam_pts.shape[0]
    A = np.zeros((2 * n, 6), dtype=np.float32)
    b = np.zeros((2 * n,), dtype=np.float32)

    for i, (cx, cy) in enumerate(cam_pts):
        sx, sy = scr_pts[i]
        A[2 * i] = [cx, cy, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, cx, cy, 1]
        b[2 * i] = sx
        b[2 * i + 1] = sy

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = params.reshape(2, 3)
    return M


def apply_affine(M: np.ndarray, cam_point: Optional[CamPoint]) -> Optional[ScreenPoint]:
    if cam_point is None:
        return None
    cx, cy = cam_point
    v = np.array([cx, cy, 1.0], dtype=np.float32)
    sx, sy = M @ v
    return int(sx), int(sy)
