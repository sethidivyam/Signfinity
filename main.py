import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset_path = 'home/hand_sign_landmarks.csv'
dataset = pd.read_csv(dataset_path)

# Prepare the data
X = dataset.iloc[:, 1:].values  # Features
y = dataset.iloc[:, 0].values   # Labels

# Replace numerical labels with words
label_map = {
    1: "No",
    2: "Sorry",
    3: "Thanks",
    4: "Yes",
    5: "Hello"
}
y = np.array([label_map.get(label, "Unknown") for label in y])

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_pca, y)

# Use the best estimator
best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Initialize Mediapipe Hands with higher thresholds for better accuracy
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Function to recognize gesture with confidence
def recognize_gesture_and_confidence(landmarks):
    landmarks = np.array(landmarks).flatten().reshape(1, -1)
    
    # Apply scaling and PCA to the new landmarks
    landmarks_scaled = scaler.transform(landmarks)
    landmarks_pca = pca.transform(landmarks_scaled)
    
    proba = best_rf.predict_proba(landmarks_pca)[0]
    gesture = best_rf.predict(landmarks_pca)[0]
    confidence = np.max(proba)
    return gesture, confidence

# Initialize webcam
cap = cv2.VideoCapture(0)

font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font_thickness = 2

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from webcam")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # Recognize gesture
            gesture, confidence = recognize_gesture_and_confidence(landmarks)

            print(f"Recognized gesture: {gesture}, Confidence: {confidence:.2f}")

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Draw bounding box
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

            # Display gesture with confidence
            text = f"{gesture} ({confidence:.2f})"
            text_x = x_min
            text_y = y_min - 10

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i != 0 or j != 0:
                        cv2.putText(frame, text, (text_x + i, text_y + j), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    else:
        print("No hand landmarks detected")

    # Display frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on Enter key
    if cv2.waitKey(1) == 13:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
