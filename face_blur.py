import cv2
import sys
import os


def pixelate_face(face_roi, pixel_size=8):
    """
    Strong pixelation effect by heavy downscaling and upscaling.
    Smaller pixel_size = stronger pixelation.
    """
    h, w = face_roi.shape[:2]

    # Shrink image drastically
    face_small = cv2.resize(
        face_roi,
        (pixel_size, pixel_size),
        interpolation=cv2.INTER_LINEAR
    )

    # Blow it back up with nearest-neighbor (blocky look)
    face_pixelated = cv2.resize(
        face_small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return face_pixelated


def blur_faces_in_image(input_path, output_path, mode="blur", blur_strength=55):
    """
    Blur or pixelate all detected faces in a single image and save result.
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Could not read image: {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # Ensure blur kernel is odd
    if blur_strength % 2 == 0:
        blur_strength += 1

    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]

        if mode == "pixelate":
            face_roi = pixelate_face(face_roi, pixel_size=6)  # VERY strong
        else:
            face_roi = cv2.GaussianBlur(
                face_roi,
                (blur_strength, blur_strength),
                0
            )

        img[y:y + h, x:x + w] = face_roi

    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved result to '{output_path}'.")


def process_path(input_path, output_path, mode):
    """
    If input_path is a file: process one image.
    If input_path is a folder: process all images.
    """
    if os.path.isdir(input_path):
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
            blur_faces_in_image(in_file, out_file, mode=mode)

        print("[INFO] Done processing folder.")
    else:
        blur_faces_in_image(input_path, output_path, mode=mode)


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python face_blur.py input_path output_path mode")
        print("")
        print("Modes:")
        print("  blur       = Gaussian blur")
        print("  pixelate   = Strong pixelation")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    mode = sys.argv[3].lower()

    if mode not in ["blur", "pixelate"]:
        print("[ERROR] Mode must be either 'blur' or 'pixelate'")
        sys.exit(1)

    process_path(input_path, output_path, mode)


if __name__ == "__main__":
    main()
    