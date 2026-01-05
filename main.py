
import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tkinter import Tk, filedialog

# 1️⃣ Khởi tạo models
yolo_bienso = YOLO("Bienso.pt")
# Thử dùng lang='ch' để đọc số chuẩn hơn nếu 'en' vẫn lỗi
ocr = PaddleOCR(lang='ch', use_textline_orientation=False)


def select_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title="Chọn ảnh biển số xe")
    root.destroy()
    return path


# 2️⃣ Chọn ảnh
image_path = select_file()
if not image_path:
    print("❌ Chưa chọn ảnh.");
    exit()

img = cv2.imread(image_path)
results = yolo_bienso(img)[0]

if len(results.boxes.xyxy) == 0:
    print("❌ YOLO không tìm thấy biển số.")
else:
    for idx, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Cắt ảnh từ YOLO
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        # --- BÍ KÍP 1: CẮT RÌA (MARGIN CROP) ---
        # Loại bỏ 10% mỗi cạnh để xóa khung đen và lưới tản nhiệt dính vào
        h_c, w_c = crop.shape[:2]
        margin_h = int(h_c * 0.1)
        margin_w = int(w_c * 0.05)
        crop = crop[margin_h:h_c - margin_h, margin_w:w_c - margin_w]

        # --- BÍ KÍP 2: TIỀN XỬ LÝ NHIỄU ---
        # Phóng to ảnh
        crop_res = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(crop_res, cv2.COLOR_BGR2GRAY)

        # Khử nhiễu làm mịn nền
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Tăng tương phản (Chỉnh clipLimit từ 10.0 - 40.0 là tối đa)
        clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Nhị phân hóa Otsu để lấy chữ đen trên nền trắng tinh
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Làm dày nét chữ một chút để AI dễ đọc (Morphology)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)

        # Chuyển về 3 kênh cho AI
        ocr_input = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # HIỂN THỊ ẢNH DEBUG
        cv2.imshow(f"AI Vision - Da loc nhieu {idx + 1}", ocr_input)

        # 3️⃣ OCR NHẬN DIỆN
        prediction = ocr.predict(ocr_input)
        ocr_rs = list(prediction)

        plate_text = ""
        if ocr_rs and ocr_rs[0]:
            raw_text = "".join([line[1][0] for line in ocr_rs[0]])

            # Làm sạch bằng Regex
            plate_text = re.sub(r'[^0-9A-Z]', '', raw_text.upper())

            # Loại bỏ các từ "ma" thường gặp
            for noise in ["NAO", "TO", "EEEE", "IEE", "NONE"]:
                plate_text = plate_text.replace(noise, "")

            # Biển số VN thường dài 7-9 ký tự, cắt bỏ nếu quá dài
            plate_text = plate_text[:10]

            print(f"✅ Biển số: {plate_text} (Gốc: {raw_text})")

            # Vẽ lên ảnh gốc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 4️⃣ Hiển thị kết quả
cv2.imshow("Final Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# 1️⃣ Khởi tạo models
yolo_bienso = YOLO("Bienso.pt")
# Dùng lang='ch' để tránh lỗi EEEE tốt hơn
ocr = PaddleOCR(lang='ch', use_textline_orientation=False)

# 2️⃣ Mở Camera (0 thường là webcam mặc định)
cap = cv2.VideoCapture(0)

# Cấu hình độ phân giải camera (Tùy chọn)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("--- Đang mở Camera... Nhấn 'q' để thoát ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3️⃣ YOLO Detect trên từng khung hình
    # imgsz=416 giúp tăng tốc độ xử lý real-time
    results = yolo_bienso(frame, conf=0.5, imgsz=416)[0]

    for idx, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Cắt ảnh biển số
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        # --- XỬ LÝ NHIỄU (Tối ưu cho Camera) ---
        h_c, w_c = crop.shape[:2]
        margin_h = int(h_c * 0.1)
        margin_w = int(w_c * 0.05)
        crop = crop[margin_h:h_c - margin_h, margin_w:w_c - margin_w]

        # Phóng to và xử lý Đen - Trắng
        crop_res = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(crop_res, cv2.COLOR_BGR2GRAY)
        
        # Tăng tương phản nhanh
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        ocr_input = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 4️⃣ OCR NHẬN DIỆN
        # Lưu ý: OCR tốn tài nguyên, trong thực tế có thể dùng luồng (threading) 
        # nhưng ở đây dùng trực tiếp để bạn dễ hiểu
        prediction = ocr.predict(ocr_input)
        ocr_rs = list(prediction)

        plate_text = ""
        if ocr_rs and ocr_rs[0]:
            raw_text = "".join([line[1][0] for line in ocr_rs[0]])
            plate_text = re.sub(r'[^0-9A-Z]', '', raw_text.upper())
            
            # Lọc nhiễu
            for noise in ["NAO", "TO", "EEEE", "IEE", "NONE"]:
                plate_text = plate_text.replace(noise, "")

            # Vẽ khung và chữ lên khung hình Camera
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"LP: {plate_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # In kết quả ra console
            if plate_text:
                print(f"📷 Camera Detect: {plate_text}")

    # 5️⃣ Hiển thị khung hình Camera
    cv2.imshow("Nhan dien Bien so Real-time", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''