# modules/metrics.py
import time
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

@dataclass
class MetricsTracker:
    total_pinch_events: int = 0
    pinch_with_active_region: int = 0

    total_summaries: int = 0
    summary_latencies_ms: List[float] = field(default_factory=list)

    total_swipe_up: int = 0
    total_swipe_down: int = 0
    
    calibration_errors_px: List[float] = field(default_factory=list)

    start_time: float = field(default_factory=time.time)

    def log_pinch(self, pinch_region_index: Optional[int], active_region_index: Optional[int]):
        self.total_pinch_events += 1
        if pinch_region_index is not None and active_region_index is not None:
            if pinch_region_index == active_region_index:
                self.pinch_with_active_region += 1

    def log_summary_latency(self, latency_ms: float):
        self.total_summaries += 1
        self.summary_latencies_ms.append(latency_ms)

    def log_swipe(self, direction: str):
        if direction == "up":
            self.total_swipe_up += 1
        elif direction == "down":
            self.total_swipe_down += 1
    def log_calibration_errors(self, errors_px: List[float]):
        self.calibration_errors_px.extend(errors_px)
    def get_runtime(self) -> float:
        return time.time() - self.start_time

    def report(self) -> str:
        runtime = self.get_runtime()
        if self.total_pinch_events > 0:
            pinch_align_rate = self.pinch_with_active_region / self.total_pinch_events
        else:
            pinch_align_rate = 0.0

        if self.summary_latencies_ms:
            avg_latency = sum(self.summary_latencies_ms) / len(self.summary_latencies_ms)
        else:
            avg_latency = 0.0
            
        if self.calibration_errors_px:
                    n = len(self.calibration_errors_px)
                    avg_err = sum(self.calibration_errors_px) / n
                    max_err = max(self.calibration_errors_px)
        else:
            avg_err = 0.0
            max_err = 0.0

        lines = [
            "Metrics Report",
            f"Runtime: {runtime:.1f} s",
            f"Total pinch events: {self.total_pinch_events}",
            f"Pinch events where active region matched: {self.pinch_with_active_region}",
            f"Gaze alignment rate (pinch vs active): {pinch_align_rate:.2f}",
            f"Total summaries triggered: {self.total_summaries}",
            f"Average summary latency: {avg_latency:.1f} ms",
            f"Total swipe up: {self.total_swipe_up}",
            f"Total swipe down: {self.total_swipe_down}",
            f"Calibration samples: {len(self.calibration_errors_px)}",
            f"Calibration error (avg pixels): {avg_err:.1f}",
            f"Calibration error (max pixels): {max_err:.1f}",
            "==========================",
        ]
        return "\n".join(lines)


@dataclass
class GestureMetricsTracker(MetricsTracker):  # Extend existing
    gesture_gt = defaultdict(list) 
    gesture_pred = defaultdict(list)

    def log_gesture(self, gesture: str, is_true: int, is_detected: int):
        self.gesture_gt[gesture].append(is_true)
        self.gesture_pred[gesture].append(is_detected)

    def compute_metrics(self) -> dict:
        results = {}
        for gesture in self.gesture_gt:
            y_true = self.gesture_gt[gesture]
            y_pred = self.gesture_pred[gesture]
            if sum(y_true) == 0:
                results[gesture] = {'precision': 0, 'recall': 0, 'f1': 0}
            else:
                p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                results[gesture] = {'precision': p, 'recall': r, 'f1': f1}
        # Macro avg across gestures
        all_p = [res['precision'] for res in results.values()]
        all_r = [res['recall'] for res in results.values()]
        all_f1 = [res['f1'] for res in results.values()]
        results['macro_avg'] = {'precision': sum(all_p)/len(all_p), 'recall': sum(all_r)/len(all_r), 'f1': sum(all_f1)/len(all_f1)}
        return results

    def report(self) -> str:
        # Extend existing report
        metrics = self.compute_metrics()
        lines = super().report().split('\n') 
        lines.append("Gesture Reliability:")
        for g, res in metrics.items():
            if g != 'macro_avg':
                lines.append(f"  {g}: P={res['precision']:.3f}, R={res['recall']:.3f}, F1={res['f1']:.3f}")
        if 'macro_avg' in metrics:
            lines.append(f"  Macro Avg: P={metrics['macro_avg']['precision']:.3f}, R={metrics['macro_avg']['recall']:.3f}, F1={metrics['macro_avg']['f1']:.3f}")
        return '\n'.join(lines)