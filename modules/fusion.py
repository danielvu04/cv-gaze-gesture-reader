from dataclasses import dataclass
from typing import List, Tuple, Optional

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

@dataclass
class Region:
    bbox: BBox
    text: str = ""
    summary: str = ""


class FusionEngine:
    """
    Handles:
      - which paragraph region the gaze is on
      - when a trigger (e.g. thumbs-up) should cause OCR + summarization
    """

    def __init__(self, regions: List[Region]):
        self.regions = regions
        self.active_index: Optional[int] = None

    def update_gaze(self, gaze_point):
        """
        Update active region index based on gaze_point (x, y).
        """
        if gaze_point is None:
            self.active_index = None
            return

        x, y = gaze_point
        self.active_index = None
        for i, region in enumerate(self.regions):
            x1, y1, x2, y2 = region.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.active_index = i
                break

    def should_trigger_summary(self, trigger: bool) -> Optional[int]:
        """
        Returns region index if a summary should be computed now.
        """
        if trigger and self.active_index is not None:
            return self.active_index
        return None
