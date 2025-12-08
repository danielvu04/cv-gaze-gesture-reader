import cv2
import pytesseract
import os
import openai 
from openai import OpenAI
from collections import Counter
import re

def extract_text_from_region(frame, bbox):
    """
    bbox: (x1, y1, x2, y2) in image coordinates
    Returns OCR string with enhanced preprocessing for gradients/colors.
    """
    # Validate frame
    if frame is None:
        return ""
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ""

    # Split channels and use the one with best contrast (for colored text)
    b, g, r = cv2.split(roi)
    gray = cv2.max(cv2.max(b, g), r)  # Light text on dark? Or invert if needed

    # CLAHE for contrast enhancement (good for gradients)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive thresholding for varying backgrounds
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Morphology to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Scale up for better resolution
    scale = 2.0 if thresh.shape[0] < 100 else 1.0
    thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Tesseract config: PSM 6 for block of text, OEM 3 for LSTM
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

def summarize_text(text):
    """
    Summarize using OpenAI with stricter prompt, or advanced extractive fallback.
    Skip if text too short.
    """
    if not text or len(text) < 50:
        return text  # Raw for short text (e.g., titles)

    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return _openai_summarize(text, api_key)
    else:
        print("Warning: OPENAI_API_KEY not set. Using extractive fallback.")
        return _extractive_summarize(text)

def _openai_summarize(text, api_key):
    try:
        client = OpenAI(api_key=api_key)
        
        if len(text) > 500:
            chunks = _chunk_text(text)
            chunk_summaries = [_openai_summarize(chunk, api_key) for chunk in chunks]
            text = " ".join(chunk_summaries)
        
        messages = [
            {"role": "system", "content": "You are a precise summarizer. Create a 1-2 sentence summary that sticks strictly to the provided text's content. Focus on core ideas only—no additions, opinions, or external info."},
            {"role": "user", "content": f"Summarize this text concisely:\n\n{text}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=80,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"OpenAI failed: {e}. Using fallback.")
        return _extractive_summarize(text)

def _chunk_text(text, max_chars=1000):
    """Chunk by approximate char length."""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i+max_chars])
    return chunks

def _extractive_summarize(text, num_sentences=2):
    """Advanced extractive: Score sentences by word frequency (normalized)."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:200] + '...' if len(text) > 200 else text

    # Word frequency (ignore common)
    common = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}
    words = re.findall(r'\w+', text.lower())
    freq = Counter(w for w in words if w not in common)

    # Score sentences
    scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\w+', sent.lower())
        score = sum(freq[w] for w in sent_words)
        scores[i] = score / max(1, len(sent_words))  # Normalize

    # Top sentences in original order
    top_idx = sorted(sorted(scores, key=scores.get, reverse=True)[:num_sentences])
    return ' '.join(sentences[i] for i in top_idx)