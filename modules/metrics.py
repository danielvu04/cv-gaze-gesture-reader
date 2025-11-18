# modules/metrics.py
import time
from dataclasses import dataclass, field
from typing import List, Optional


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
