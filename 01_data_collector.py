import cv2
import os

# --- 1. SETUP ---
# Define the folder where images will be stored
image_dir = "ImagesAttendance"

# Enter the ID and Name for the new person
# IMPORTANT: Use a unique ID number (like 1, 2, 3) and a simple name (e.g., Alice).
face_id = input('Enter Person ID (e.g., 1): ')
name = input('Enter Person Name (e.g., Manya): ')

# Use the default webcam (usually 0). Check if the camera is available.
cam = cv2.VideoCapture(0) 
if not cam.isOpened():
    print("Error: Could not open webcam. Check if another program is using it.")
    exit()

cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Use a pre-trained Haar Cascade model for quick face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n[INFO] Initializing face capture. Look directly at the camera...")

# Initialize individual sampling face count
count = 0

# --- 2. CAPTURE LOOP ---
while(True):
    # Read frame from the camera
    ret, img = cam.read()
    # Flip the image horizontally (mirror effect)
    img = cv2.flip(img, 1) 
    # Convert image to grayscale for better detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_detector.detectMultiScale(
        gray, 
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x,y,w,h) in faces:
        # Draw a blue rectangle around the detected face
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        
        count += 1

        # Save the captured face image (cropped to the face area)
        # Filename format: ID.Name.Count.jpg
        file_name = f"{face_id}.{name}.{count}.jpg"
        cv2.imwrite(os.path.join(image_dir, file_name), gray[y:y+h,x:x+w])

        # Display the live video feed
        cv2.imshow('Image Collector - Press ESC to exit', img)

    # --- 3. BREAK CONDITIONS ---
    # Press 'ESC' key (ASCII 27) to exit the video feed early
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    # Stop collecting after 30 samples
    elif count >= 30: 
         break

# --- 4. CLEAN UP ---
print(f"\n[INFO] {count} samples collected for {name}. Exiting Program and cleaning up resources.")
cam.release()
cv2.destroyAllWindows()