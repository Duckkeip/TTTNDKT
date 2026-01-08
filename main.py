import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tkinter import Tk, filedialog

# 1. Khởi tạo
yolo_model = YOLO("Bienso.pt")
# Khởi tạo PaddleOCR với cấu hình ổn định
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def select_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title="Chọn ảnh biển số xe")
    root.destroy()
    return path


image_path = select_file()
if not image_path: exit()

img = cv2.imread(image_path)
results = yolo_model(img)[0]

if len(results.boxes.xyxy) == 0:
    print("❌ YOLO không tìm thấy biển số.")
else:
    for idx, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Cắt biển số (nới rộng 2px để tránh mất nét)
        crop = img[max(0, y1 - 2):min(img.shape[0], y2 + 2),
        max(0, x1 - 2):min(img.shape[1], x2 + 2)]
        if crop.size == 0: continue

        # --- TIỀN XỬ LÝ ---
        h, w = crop.shape[:2]
        crop = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        ocr_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # --- NHẬN DIỆN (SỬA LỖI TẠI ĐÂY) ---
        # Sử dụng ocr.ocr() nhưng truy xuất theo cấu trúc an toàn
        output = ocr.ocr(ocr_input)

        plate_text = ""
        if output and output[0]:
            for line in output[0]:
                # Kiểm tra chắc chắn line có cấu trúc: [ [box], (text, confidence) ]
                if isinstance(line, list) and len(line) > 1:
                    data = line[1]  # Lấy phần (text, confidence)
                    if isinstance(data, tuple) and len(data) > 1:
                        text = data[0]
                        conf = data[1]

                        if conf > 0.4:
                            print(f"--- AI thấy: {text} ({round(conf, 2)})")
                            plate_text += text

            # Làm sạch kết quả
            plate_text = re.sub(r'[^0-9A-Z]', '', plate_text.upper())
            # Sửa lỗi nhận diện nhầm ký tự thường gặp
            mapping = {'S': '5', 'G': '6', 'O': '0', 'D': '0'}
            for char, replace_char in mapping.items():
                # Chỉ thay thế ở những vị trí hợp lý (ví dụ đầu biển thường là số)
                if plate_text.startswith(char):
                    plate_text = replace_char + plate_text[1:]

            print(f"✅ KẾT QUẢ CUỐI: {plate_text}")

            # Vẽ lên ảnh
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Result", img)
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