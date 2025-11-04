import face_recognition
import cv2
import numpy as np
import os
import pickle

# --- 1. SETUP & DATA PATHS ---
# Path to the folder containing training images
image_dir = "ImagesAttendance"

# File name for saving the final encoded data
# This file stores the AI's "memory"
encoding_file = "face_encodings.pickle"

# Initialize list to hold encoded data and names
known_face_encodings = []
known_face_names = []

# --- 2. ENCODING LOOP ---
print("[INFO] Starting to encode faces from dataset...")

# Iterate through every file in the image directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # We extract the name from the filename (e.g., "Manya")
        try:
            # NOTE: Assumes filename is like ID.Name.Count.jpg
            name = filename.split(".")[1] 
        except IndexError:
            # Skip files with unexpected names
            print(f"[WARNING] Skipping file with unexpected name format: {filename}")
            continue
        
        # Construct the full path to the image
        image_path = os.path.join(image_dir, filename)
        
        # Load the image using face_recognition library
        image = face_recognition.load_image_file(image_path)
        
        # Find the face locations in the image 
        face_locations = face_recognition.face_locations(image)
        
        # If a face is found, compute the 128-dimensional face encoding
        if face_locations:
            # Use the first face found in the image
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            # Add the encoding and name to our lists
            known_face_encodings.append(encoding)
            known_face_names.append(name)
        else:
            print(f"[WARNING] No face found in {filename}. Skipping.")


# --- 3. SAVE ENCODINGS ---
print("[INFO] Encoding complete. Saving data...")

# Create a dictionary to hold the data
data = {"encodings": known_face_encodings, "names": known_face_names}

# Use the pickle library to serialize (save) the data to a file
with open(encoding_file, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] Successfully saved {len(known_face_encodings)} face samples into {encoding_file}!")