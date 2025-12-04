import cv2
import pytesseract
import os
import openai
from openai import OpenAI
import re  

def extract_text_from_region(frame, bbox):
    """
    bbox: (x1, y1, x2, y2) in image coordinates
    Returns OCR string with improved preprocessing.
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    if gray.shape[0] < 50 or gray.shape[1] < 100:
        scale = max(50 / gray.shape[0], 100 / gray.shape[1], 1.0)
        new_h, new_w = int(gray.shape[0] * scale), int(gray.shape[1] * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    config = "--psm 7"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

def summarize_text(text):
    """
    Summarize using OpenAI GPT-3.5-turbo with improved prompt, or fallback to extractive summary.
    """
    if not text or len(text) < 50:
        return text[:150] + "..." if len(text) > 150 else text

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Using extractive fallback.")
        return _extractive_fallback(text)

    try:
        client = OpenAI(api_key=api_key)
        
        if len(text) > 500:
            chunks = _chunk_text(text, max_words=200)
            chunk_summaries = [summarize_text(chunk) for chunk in chunks]  
            text = " ".join(chunk_summaries)  
        
        messages = [
            {"role": "system", "content": "You are a concise reading assistant. Summarize text in exactly 2-3 sentences, focusing only on the key ideas and main points. Use clear, simple language. Do not add, invent, or assume any information not explicitly in the text."},
            {"role": "user", "content": f"Summarize the following text, highlighting the core message and supporting facts:\n\n{text}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=120,
            temperature=0.1 
        )
        summary = response.choices[0].message.content.strip()
        return summary if len(summary) < 300 else summary[:300] + "..."
    
    except Exception as e:
        print(f"OpenAI summarization failed: {e}. Using extractive fallback.")
        return _extractive_fallback(text)

def _chunk_text(text, max_words=200):
    """Simple chunker by sentences/words."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    current_words = 0
    for sent in sentences:
        words = len(sent.split())
        if current_words + words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_words = words
        else:
            current_chunk.append(sent)
            current_words += words
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def _extractive_fallback(text, max_sentences=3):
    """Simple extractive summary: Pick top sentences by length/keywords."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    scored = sorted(sentences, key=len, reverse=True)[:max_sentences]
    return " ".join(scored).strip()[:250] + "..."