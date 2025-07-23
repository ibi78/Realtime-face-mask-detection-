import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Load the trained Keras model ===
model = load_model("face_mask_detector.keras")

# === Set input size based on your training setup ===
IMG_SIZE = 150  # Use 224 if your model was trained on 224x224

# === Load Haar Cascade face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Open webcam ===
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess the face ROI
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # === Predict ===
        prediction = model.predict(face)[0][0]  # single output (sigmoid)

        # ✅ Debug print statements
        print(f"Prediction raw value: {prediction:.4f}")

        # Determine label and color
        label = "No Mask" if prediction >= 0.5 else "Mask"
        confidence = prediction if label == "No Mask" else 1 - prediction
        color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)

        print(f"Label: {label}, Confidence: {confidence * 100:.2f}%")

        # Draw label and bounding box
        cv2.putText(frame, f"{label}: {confidence * 100:.2f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show the frame
    cv2.imshow("Live Face Mask Detection", frame)

    # Break the loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
