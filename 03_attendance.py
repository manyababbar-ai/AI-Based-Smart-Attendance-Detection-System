# ...existing code...
import face_recognition
import cv2
import time
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
import sys

# --- 1. SETUP ---
# File containing the AI's "memory" (encodings)
encoding_file = "face_encodings.pickle"

# File where attendance will be logged
attendance_log_file = "Attendance_Log.csv"

# Load the stored face encodings and names
print("[INFO] Loading face encodings...")
try:
    with open(encoding_file, "rb") as f:
        data = pickle.loads(f.read())
        known_face_encodings = data.get("encodings", [])
        known_face_names = data.get("names", [])
        if not isinstance(known_face_encodings, list) or not isinstance(known_face_names, list):
            raise ValueError("Invalid format in encoding file.")
except Exception as e:
    print(f"[WARN] Could not load encodings ({e}). Continuing with empty known faces.")
    known_face_encodings = []
    known_face_names = []

# List to keep track of students already marked present
students_present = []

# --- 2. WEBCAM SETUP ---
# Try multiple indices and backends (Windows common backends)
def try_open_camera(indices=(0,1,2,3), backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)):
    for backend in backends:
        for idx in indices:
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                # fallback for OpenCV versions that don't accept backend in constructor
                try:
                    cap = cv2.VideoCapture(idx)
                except Exception:
                    continue
            time.sleep(0.3)
            if cap.isOpened():
                print(f"[INFO] Opened camera index={idx} backend={backend}")
                return cap
            cap.release()
    return None

video_capture = try_open_camera()

# Check if webcam is available
if video_capture is None or not video_capture.isOpened():
    print("Error: Could not open webcam. Try closing other apps, check Camera privacy (Settings > Privacy > Camera), or try different camera index.")
    sys.exit(1)

print("[INFO] Attendance system is LIVE! Press 'q' to stop.")

# --- 3. LOGIC LOOP ---
try:
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print("[WARN] Failed to read frame from camera. Retrying in 0.2s...")
            time.sleep(0.2)
            continue

        # Flip image for a mirror view (makes recognition easier)
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each face found in the current frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # --- RECOGNITION ---
            name = "Unknown"

            if known_face_encodings:
                # Compare the current face to all known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                # Find the best match (the shortest distance)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                # If the best match is actually a match (distance is small enough)
                if matches and matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]

                    # --- ATTENDANCE LOGGING ---
                    if name not in students_present:
                        now = datetime.now()
                        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

                        # Prepare the log data
                        log_entry = pd.DataFrame([{'Name': name, 'Time': dt_string, 'Status': 'Present'}])

                        # Check if file exists to avoid writing header multiple times
                        if not os.path.exists(attendance_log_file):
                            log_entry.to_csv(attendance_log_file, mode='a', header=True, index=False)
                        else:
                            log_entry.to_csv(attendance_log_file, mode='a', header=False, index=False)

                        students_present.append(name)
                        print(f"[ATTENDANCE] Marked PRESENT for: {name} at {dt_string}")
            else:
                # No known encodings loaded
                # Optionally display a warning once per session
                if not students_present:
                    print("[WARN] No known face encodings loaded — system will show all faces as Unknown.")

            # --- DISPLAY ---
            # Draw a box and label around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('AI Smart Attendance System', frame)

        # Stop condition: Hit 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
finally:
    # --- 4. CLEAN UP ---
    print("\n[INFO] Shutting down system and cleaning up resources.")
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows()
# ...existing code...