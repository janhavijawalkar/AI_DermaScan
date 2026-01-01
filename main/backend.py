import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


# Load trained model

model = load_model(
    r"C:\Users\user\OneDrive\Desktop\DermalScan\main\best_balanced1_noaug.keras"
)

# Class labels
labels = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Dynamic label 

def draw_label(img, text, confidence):
    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Slightly bigger, professional size
    font_scale = max(0.8, w / 850)
    thickness = int(max(2, w / 480))

    label = f"{text}  {confidence:.1f}%"

    # Always keep text inside image
    x = 12
    y = int(35 + font_scale * 10)

    cv2.putText(
        img,
        label,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),   
        thickness,
        cv2.LINE_AA
    )

    return img



# Preprocess face

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (224, 224))
    face_input = preprocess_input(face_resized.astype(np.float32))
    return np.expand_dims(face_input, axis=0)


# Prediction

def predict_skin_condition(face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)[0]

    return {
        "label": labels[int(np.argmax(preds))],
        "confidence": float(np.max(preds) * 100),
        "probabilities": {
            labels[i]: float(preds[i] * 100)
            for i in range(len(labels))
        }
    }


# Main pipeline

def process_image(img):
    start_time = time.time()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(60, 60)
    )

    annotated_img = img.copy()
    results = []

    
    # CASE 1: No face detected
    
    if len(faces) == 0:
        output = predict_skin_condition(img)
        annotated_img = draw_label(
            annotated_img,
            output["label"],
            output["confidence"]
        )
        results.append(output)

    
    # CASE 2: Face(s) detected
    
    else:
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            output = predict_skin_condition(face_img)

            cv2.rectangle(
                annotated_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            annotated_img = draw_label(
                annotated_img,
                output["label"],
                output["confidence"]
            )

            results.append(output)

    print("\n--- Prediction Log ---")
    print(f"Faces detected: {len(faces)}")
    print(f"Processing time: {time.time() - start_time:.2f} sec")

    return annotated_img, results
