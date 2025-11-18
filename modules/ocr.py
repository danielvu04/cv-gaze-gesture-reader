import cv2
import pytesseract


def extract_text_from_region(frame, bbox):
    """
    bbox: (x1, y1, x2, y2) in image coordinates
    Returns OCR string.
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # basic preprocessing, you can tune this
    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = "--psm 6"
    text = pytesseract.image_to_string(gray, config=config)
    return text.strip()


def summarize_text(text):
    """
    Placeholder for GPT based summarization.
    Right now it just truncates or returns original text.
    Replace this with an API call later.
    """
    if not text:
        return ""

    return text
