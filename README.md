# ✋ Signfinity – Real-Time Hand Gesture Recognition

**Signfinity** is a real-time hand gesture recognition system using a webcam to identify five predefined gestures: "Hello", "Yes", "No", "Thanks", and "Sorry". It leverages **MediaPipe** for hand landmark detection and a machine learning pipeline involving **PCA** and **Random Forest Classifier** to make accurate predictions in real time.

---

## 🧠 How It Works

1. Captures hand landmarks (21 points) in 3D using **MediaPipe Hands**
2. Flattens and scales the data using **StandardScaler**
3. Applies **PCA** to reduce dimensions while retaining 95% of variance
4. Trains a **Random Forest Classifier** (with GridSearch hyperparameter tuning)
5. Uses **OpenCV** to display predictions in real-time along with confidence scores

---

## 📁 Dataset

- File: `signfinity/hand_sign_landmarks.csv`
- Format:
  - Column 0: Label (1 to 5)
  - Columns 1–63: Flattened (x, y, z) coordinates from 21 landmarks

### Label Mapping

| Label | Gesture |
|-------|---------|
| 1     | No      |
| 2     | Sorry   |
| 3     | Thanks  |
| 4     | Yes     |
| 5     | Hello   |

---

## 📦 Requirements

Install dependencies with pip:

```bash
pip install opencv-python mediapipe scikit-learn numpy pandas
