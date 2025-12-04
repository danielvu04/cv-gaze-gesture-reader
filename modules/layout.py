from typing import List
import cv2
import pytesseract
from PIL import Image
from modules.fusion import Region

def detect_text_regions(frame, min_conf=30, left_margin=0.5, max_regions=5):
    """
    Detect adaptive text regions based on OCR word bounding boxes.
    Clusters words -> lines -> paragraphs using vertical density/gaps.
    
    Args:
        frame: Screen capture (BGR numpy array).
        min_conf: Min Tesseract confidence for words.
        left_margin: Fraction of width to consider "reading area" (default: left half).
        max_regions: Cap number of regions (for UI stability).
    
    Returns:
        List[Region] with bbox and empty text/summary.
    """
    h, w, _ = frame.shape
    reading_width = int(w * left_margin)
    
    # Convert to PIL for Tesseract
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Get word-level data
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config='--psm 6')
    
    # Filter valid words (conf > min_conf, non-empty text)
    words = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > min_conf and data['text'][i].strip():
            x, y, wd, ht = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Limit to reading area
            if x < reading_width:
                words.append((x, y, wd, ht, data['text'][i]))
    
    if not words:
        # Fallback to original fixed regions if no text detected
        return create_fallback_regions(w, h)
    
    # Sort words by y (top-to-bottom), then x (left-to-right)
    words.sort(key=lambda b: (b[1], b[0]))
    
    # Group into lines (words with y-diff < 20px)
    lines = []
    if words:
        current_line = [words[0]]
        line_y = words[0][1]
        for word in words[1:]:
            if abs(word[1] - line_y) < 20:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                line_y = word[1]
        lines.append(current_line)
    
    # Group lines into paragraphs (gap > 1.5 * avg line height)
    if not lines:
        return []
    
    line_heights = [max(b[3] for b in line) for line in lines]
    avg_h = sum(line_heights) / len(line_heights)
    para_gap = avg_h * 1.5
    
    paragraphs = []
    current_para = [lines[0]]
    prev_bottom = min(b[1] for b in lines[0]) + max(b[3] for b in lines[0])
    
    for line_group in lines[1:]:
        top_y = min(b[1] for b in line_group)
        if top_y - prev_bottom > para_gap:
            # Compute bbox and text for current para
            min_x = min(min(b[0] for b in line) for line in current_para)
            max_x = max(max(b[0] + b[2] for b in line) for line in current_para)
            min_y = min(min(b[1] for b in line) for line in current_para)
            max_y = max(max(b[1] + b[3] for b in line) for line in current_para)
            bbox = (min_x, min_y, min(max_x, reading_width), max_y)
            
            # Create region (text extracted later)
            paragraphs.append(Region(bbox=bbox, text="", summary=""))
            
            current_para = [line_group]
        else:
            current_para.append(line_group)
        prev_bottom = min(b[1] for b in line_group) + max(b[3] for b in line_group)
    
    # Last paragraph
    if current_para:
        min_x = min(min(b[0] for b in line) for line in current_para)
        max_x = max(max(b[0] + b[2] for b in line) for line in current_para)
        min_y = min(min(b[1] for b in line) for line in current_para)
        max_y = max(max(b[1] + b[3] for b in line) for line in current_para)
        bbox = (min_x, min_y, min(max_x, reading_width), max_y)
        paragraphs.append(Region(bbox=bbox, text="", summary=""))
    
    # Cap regions and return
    return paragraphs[:max_regions]

def create_fallback_regions(width: int, height: int) -> List[Region]:
    """Original fixed regions as fallback."""
    margin = 50
    col_width = int(width * 0.45)
    row_height = int((height - 2 * margin) / 3)
    regions = []
    for i in range(3):
        x1 = margin
        y1 = margin + i * row_height
        x2 = x1 + col_width
        y2 = y1 + row_height - 20
        regions.append(Region(bbox=(x1, y1, x2, y2)))
    return regions