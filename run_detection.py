import cv2
import tensorflow as tf
import numpy as np
import os
import winsound

# --- Configuration ---
MODEL_PATH = 'my_model.h5'
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
# Using eye_tree_eyeglasses for better detection if user wears glasses
EYE_CASCADE_PATH = 'haarcascade_eye_tree_eyeglasses.xml'

# Alarm settings
FREQUENCY = 2500 # Hz
DURATION = 1000  # ms
THRESHOLD = 5    # Consecutive frames with closed eyes to trigger alarm

def load_resources():
    print("Loading model and cascades...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run the notebook to train and save the model.")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + EYE_CASCADE_PATH)
    
    if face_cascade.empty() or eye_cascade.empty():
        # Fallback to local files if cv2.data.haarcascades fails
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        
    if face_cascade.empty():
        raise IOError(f"Failed to load face cascade from {FACE_CASCADE_PATH}")
    if eye_cascade.empty():
        raise IOError(f"Failed to load eye cascade from {EYE_CASCADE_PATH}")
        
    return model, face_cascade, eye_cascade

def main():
    try:
        model, face_cascade, eye_cascade = load_resources()
    except Exception as e:
        print(f"Error: {e}")
        return

    cap = cv2.VideoCapture(0)  # Try default webcam
    if not cap.isOpened():
        cap = cv2.VideoCapture(1) # Try external webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting detection. Press 'q' to quit.")
    
    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
        
        status = "No Eyes Detected"
        color = (255, 255, 255) # White

        if len(eyes) > 0:
            # Process each eye (usually we only care if both are closed, but logic uses any)
            for (ex, ey, ew, eh) in eyes:
                eye_roi = frame[ey:ey+eh, ex:ex+ew]
                
                # Preprocess for model
                final_image = cv2.resize(eye_roi, (224, 224))
                final_image = np.array(final_image).reshape(1, 224, 224, 3)
                final_image = final_image / 255.0
                
                predictions = model.predict(final_image, verbose=0)
                
                if predictions[0][0] > 0.5:
                    status = "Open Eyes"
                    color = (0, 255, 0) # Green
                    counter = 0 # Reset counter if eyes are open
                else:
                    status = "Closed Eyes"
                    color = (0, 0, 255) # Red
                    counter += 1
                
                # Draw box around eye
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
                break # Only process one eye for status display to avoid flickering

        # Display status text
        cv2.putText(frame, status, (50, 50), font, 1, color, 2, cv2.LINE_AA)

        # Check Threshold
        if counter >= THRESHOLD:
            cv2.putText(frame, "SLEEP ALERT!!", (frame.shape[1]//4, frame.shape[0]//2), 
                        font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            print("Alert: Drowsiness detected!")
            winsound.Beep(FREQUENCY, DURATION)
            counter = 0 # Reset after alert

        cv2.imshow('Driver Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
