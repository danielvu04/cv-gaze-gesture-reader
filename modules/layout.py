from typing import List
from modules.fusion import Region

def create_screen_regions(frame_width: int, frame_height: int) -> List[Region]:
    """
    Define reading regions on the left half of the captured area.
    These are used to decide which paragraph the user is 'looking at'.
    """
    margin = 50
    col_width = int(frame_width * 0.45)
    row_height = int((frame_height - 2 * margin) / 3)

    regions = []
    for i in range(3):
        x1 = margin
        y1 = margin + i * row_height
        x2 = x1 + col_width
        y2 = y1 + row_height - 20
        regions.append(Region(bbox=(x1, y1, x2, y2)))
    return regions
