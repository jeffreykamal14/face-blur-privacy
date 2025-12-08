import sys
import os
import cv2


def blur_faces(input_path, output_path):
    # Check file exists
    if not os.path.exists(input_path):
        print(f"Error: Input image '{input_path}' not found.")
        return

    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image '{input_path}'.")
        return

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's built-in Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Could not load Haar cascade for face detection.")
        return

    # Detect faces (x, y, w, h for each face)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Detected {len(faces)} face(s) in the image.")

    # For each face, blur the region
    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = image[y:y + h, x:x + w]

        # Apply a blur (Gaussian blur)
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 0)

        # Put blurred face back into the image
        image[y:y + h, x:x + w] = blurred_face

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Saved blurred image to '{output_path}'.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python face_blur.py input_image output_image")
        print("Example: python face_blur.py input.jpg output_blurred.jpg")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    blur_faces(input_path, output_path)


if __name__ == "__main__":
    main()
