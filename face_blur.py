import cv2
import sys

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Read input arguments
input_image_path = sys.argv[1]
output_image_path = sys.argv[2]

# Load the image
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(f"Detected {len(faces)} face(s) in the image.")

# Apply STRONG blur to each detected face
for (x, y, w, h) in faces:
    face_region = image[y:y+h, x:x+w]

    
    face_region = cv2.GaussianBlur(face_region, (99, 99), 30)

    image[y:y+h, x:x+w] = face_region

# Save output
cv2.imwrite(output_image_path, image)
print(f"Saved blurred image to '{output_image_path}'.")
