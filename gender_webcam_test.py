import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# --- 1) Load the trained model file ---
MODEL_FILENAME = "gender_classifier_model.h5"
if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILENAME}")

model = load_model(MODEL_FILENAME)
print("Model loaded:", MODEL_FILENAME)

# --- 2) Labels (0 = Female, 1 = Male) ---
LABELS = {0: "Female", 1: "Male"}

# --- 3) Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Try changing index (0→1).")

# --- 4) Load Haar Cascade for face detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

IMG_SIZE = (150, 150)

print("Press 'q' to quit the webcam window.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, IMG_SIZE)
        except:
            continue

        face_array = face_resized.astype("float32") / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        pred = model.predict(face_array)[0]
        if len(pred) == 1:  # sigmoid output
            score = float(pred[0])
            class_idx = 1 if score > 0.5 else 0
            confidence = score if class_idx == 1 else 1 - score
        else:  # softmax output
            class_idx = int(np.argmax(pred))
            confidence = float(pred[class_idx])

        label = LABELS[class_idx]
        conf_text = f"{label} ({confidence*100:.0f}%)"

        color = (0, 255, 0) if label == "Male" else (255, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Gender Classifier (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
