import cv2
import sys
import os


def blur_faces_in_image(input_path, output_path, blur_strength=55):
    """
    Blur all detected faces in a single image and save result.
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Could not read image: {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's built-in Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"[INFO] Detected {len(faces)} face(s) in '{os.path.basename(input_path)}'.")

    # Ensure blur kernel size is odd
    if blur_strength % 2 == 0:
        blur_strength += 1

    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]

        # Stronger Gaussian blur
        face_roi = cv2.GaussianBlur(
            face_roi,
            (blur_strength, blur_strength),
            0
        )

        img[y:y + h, x:x + w] = face_roi

    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved blurred image to '{output_path}'.")


def process_path(input_path, output_path):
    """
    If input_path is a file: blur that one image.
    If input_path is a folder: blur all images inside and save them to output_path folder.
    """
    if os.path.isdir(input_path):
        # Folder mode
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

        files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(valid_ext)
        ]

        if not files:
            print(f"[WARN] No image files found in folder: {input_path}")
            return

        print(f"[INFO] Found {len(files)} image(s) in '{input_path}'. Processing...")

        for filename in files:
            in_file = os.path.join(input_path, filename)
            out_file = os.path.join(output_path, filename)
            blur_faces_in_image(in_file, out_file)

        print("[INFO] Done processing folder.")
    else:
        # Single file mode
        blur_faces_in_image(input_path, output_path)


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python face_blur.py input_path output_path")
        print("")
        print("Where:")
        print("  input_path  = single image file OR folder with images")
        print("  output_path = output image file (if input is file)")
        print("                OR folder to save blurred images (if input is folder)")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    process_path(input_path, output_path)


if __name__ == "__main__":
    main()

