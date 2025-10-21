# Smart Reading Assistant with Computer Vision

### Team Name: DJT

**Course:** CSCI 5561 – Computer Vision  
**Semester:** Fall 2025  
**Instructor:** Prof. Karthik Desingh  
**University of Minnesota – Twin Cities**

---

## Project Overview
The **Smart Reading Assistant** is a computer vision–based system that enables **hands-free and intuitive interaction with digital text** using **gaze tracking** and **hand gesture recognition**.  
By leveraging webcam input and state-of-the-art CV models, the system detects where the user is looking and recognizes specific gestures to control scrolling, highlighting, or reading focus — improving accessibility and user experience.

**Core Goals**
- Detect user **gaze direction** to identify on-screen reading focus.  
- Recognize **hand gestures** (e.g., swipe, pinch) to enable scrolling or selection.  
- Integrate both modalities for **multimodal interaction**.  
- Evaluate performance on benchmark datasets and custom test cases.  

---

## Technical Components
| Component | Description | Method / Library |
|------------|--------------|------------------|
| **Face & Eye Tracking** | Detects facial landmarks and estimates gaze vector | MediaPipe / OpenCV |
| **Hand Gesture Recognition** | Recognizes user gestures for interaction | MediaPipe Hands / OpenCV |
| **Integration Module** | Combines gaze and gesture input to control the interface | Custom fusion logic |
| **Interface Layer** | Visualizes text focus and gesture-based control | PyQt / OpenCV display |

---

## Team Members

| Name | Email | Role |
|------|--------|------|
| **Daniel Vu**    | vu000194@umn.edu | [roles] |
| **Joshua Cheng** | chen7647@umn.edu | [roles] |
| **Thang Pham**   | pham0503@umn.edu | [roles] |

---

## ⚙️ Setup Instructions
1. **Clone Repository**
   ```bash
   git clone https://github.com/danielvu04/cv-gaze-gesture-reader/.git
   cd cv-gaze-gesture-reader
