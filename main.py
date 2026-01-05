from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

# Load models
yolo_bienso = YOLO("Bienso.pt")
yolo_thesv  = YOLO("Thesv.pt")   # hiện chưa dùng, để sẵn
ocr = PaddleOCR(use_textline_orientation=True, lang='en')


# Read image
img = cv2.imread("test.png")
if img is None:
    raise ValueError("Không đọc được ảnh test.png")

# Detect biển số
results = yolo_bienso(img)[0]

for box in results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)

    # Cắt an toàn
    h, w, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # OCR
    ocr_rs = ocr.ocr(crop, cls=True)
    if ocr_rs and ocr_rs[0]:
        text = ocr_rs[0][0][1][0]
        print("Biển số:", text)
