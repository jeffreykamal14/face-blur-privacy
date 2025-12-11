# face-blur-privacy

## How It Works

1. The program loads an input image.
2. The image is converted to grayscale to improve detection speed.
3. OpenCV’s Haar Cascade face detector finds all visible faces.
4. Each detected face is blurred or pixelated.
5. The final edited image is saved to a new output folder.

face-blur-privacy demonstrates the use of:
- Image preprocessing  
- Face detection  
- Filtering (blurring & pixelation)  
- Bounding box manipulation  
- Batch image processing  


---

## Files in This Repository

- `face_blur.py` → Main Python script
- `images_raw/` → Folder containing original input images  
- `images_blurred/` → Folder containing blurred output images  
- `images_pixelated/` → Folder containing pixelated output images  
- `README.md` → Project documentation
  
---

## How to Run the Program

Blur All Images in a Folder:
python face_blur.py images_raw images_blurred

Pixelate All Images in a Folder:
python face_blur.py images_raw images_pixelated pixel

Blur a Single Image:
python face_blur.py people.jpg people_blurred.jpg

Pixelate a Single Image:
python face_blur.py people.jpg people_pixelated.jpg pixel

---

## Requirements

- Python 3
- OpenCV

Install OpenCV using:

```bash
pip install opencv-python
