# face-blur-privacy

## How It Works

1. The program loads an input image.
2. The image is converted to grayscale.
3. OpenCV’s Haar Cascade face detector finds all faces.
4. Each detected face is blurred using a heavy blur filter.
5. The final blurred image is saved as a new file.

---

## Files in This Repository

- `face_blur.py` → Main Python script
- `people.jpg` → Sample input image
- `people_blurred.jpg` → Output image with blurred faces (optional)
- `README.md` → Project explanation

---

## Requirements

- Python 3
- OpenCV

Install OpenCV using:

```bash
pip install opencv-python

---

## How to Run the Program

Run this command in your terminal:

python face_blur.py people.jpg people_blurred.jpg


After running, the blurred image will be saved as:

people_blurred.jpg