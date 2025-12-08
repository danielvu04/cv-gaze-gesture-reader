from typing import List
import cv2
import pytesseract
from PIL import Image
from modules.fusion import Region


def detect_text_regions(
    frame,
    min_conf: int = 60,
    left_margin: float = 1.0,
    max_regions: int = 5,
) -> List[Region]:
    """
    Detect adaptive text regions based on OCR word bounding boxes.
    Clusters words -> lines -> paragraphs using vertical density and gaps.

    Args:
        frame: Screen capture (BGR numpy array).
        min_conf: Min Tesseract confidence for words.
        left_margin: Fraction of width to consider "reading area"
                     (1.0 = full width, 0.5 = left half, etc).
        max_regions: Cap number of regions (for UI stability).

    Returns:
        List[Region] with bbox and empty text/summary.
    """
    h, w, _ = frame.shape
    reading_width = int(w * left_margin)

    # Convert to PIL for Tesseract
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Get word-level data
    data = pytesseract.image_to_data(
        pil_img,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )

    # Filter valid words (conf > min_conf, non-empty text)
    words = []
    n_items = len(data["text"])
    for i in range(n_items):
        try:
            conf = int(data["conf"][i])
        except ValueError:
            continue

        if conf > min_conf and data["text"][i].strip():
            x = data["left"][i]
            y = data["top"][i]
            wd = data["width"][i]
            ht = data["height"][i]

            # Restrict to reading area
            if x < reading_width:
                words.append((x, y, wd, ht, data["text"][i]))

    if not words:
        return create_fallback_regions(w, h)

    # Sort words by y (top to bottom) then x (left to right)
    words.sort(key=lambda b: (b[1], b[0]))

    # Group words into lines
    lines = []
    current_line = [words[0]]
    line_y = words[0][1]

    for word in words[1:]:
        if abs(word[1] - line_y) < 25:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
            line_y = word[1]
    lines.append(current_line)

    if not lines:
        return create_fallback_regions(w, h)

    # Compute average line height
    line_heights = [max(b[3] for b in line) for line in lines]
    avg_h = sum(line_heights) / len(line_heights)
    para_gap = avg_h * 1.5

    # Group lines into paragraphs
    paragraphs = []
    current_para = [lines[0]]
    prev_bottom = min(b[1] for b in lines[0]) + max(b[3] for b in lines[0])

    for line_group in lines[1:]:
        top_y = min(b[1] for b in line_group)
        if top_y - prev_bottom > para_gap:
            paragraphs.append(current_para)
            current_para = [line_group]
        else:
            current_para.append(line_group)
        prev_bottom = min(b[1] for b in line_group) + max(b[3] for b in line_group)

    if current_para:
        paragraphs.append(current_para)

    # Convert paragraphs to Region objects
    regions: List[Region] = []
    pad_x = 20  # horizontal padding

    for para in paragraphs[:max_regions]:
        min_x = min(min(b[0] for b in line) for line in para)
        max_x = max(max(b[0] + b[2] for b in line) for line in para)
        min_y = min(min(b[1] for b in line) for line in para)
        max_y = max(max(b[1] + b[3] for b in line) for line in para)

        x1 = max(0, min_x - pad_x)
        x2 = min(reading_width, max_x + pad_x)

        bbox = (x1, min_y, x2, max_y)
        regions.append(Region(bbox=bbox, text="", summary=""))

    # If for some reason we ended up with no valid regions, fall back
    if not regions:
        return create_fallback_regions(w, h)

    return regions


def create_fallback_regions(width: int, height: int) -> List[Region]:
    """
    Simple fixed layout: 3 horizontal bands on the left side of the screen.
    Used as a fallback when OCR does not detect any text.
    """
    margin = 50
    x1 = margin
    x2 = int(width * 0.45)
    usable_h = height - 2 * margin
    band_height = usable_h // 3

    regions: List[Region] = []
    for i in range(3):
        ry1 = margin + i * band_height
        if i == 2:
            ry2 = height - margin
        else:
            ry2 = ry1 + band_height

        bbox = (x1, ry1, x2, ry2)
        regions.append(Region(bbox=bbox, text="", summary=""))

    return regions
