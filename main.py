import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from tkinter import Tk, filedialog

# 1. Khởi tạo Models
yolo_model = YOLO("Bienso.pt")
reader = easyocr.Reader(['en'], gpu=False)


def select_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title="Chọn ảnh biển số xe")
    root.destroy()
    return path


def vietnamese_plate_correction(text):
    """Hàm sửa lỗi dựa trên logic định dạng biển số VN"""
    text = re.sub(r'[^0-9A-Z]', '', text.upper())
    if len(text) < 7: return text

    chars = list(text)
    # Quy tắc: Ký tự thứ 3 (index 2) thường là CHỮ (K, L, M, N...)
    map_to_char = {'1': 'I', '7': 'T', '0': 'O', '5': 'S', '2': 'Z'}
    if chars[2].isdigit():
        chars[2] = map_to_char.get(chars[2], chars[2])

    # Quy tắc: Ký tự thứ 4 (index 3) thường là SỐ
    map_to_num = {'I': '1', 'T': '7', 'S': '5', 'G': '6', 'B': '8', 'D': '0'}
    if not chars[3].isdigit():
        chars[3] = map_to_num.get(chars[3], chars[3])

    return "".join(chars)


image_path = select_file()
if not image_path: exit()

img = cv2.imread(image_path)
results = yolo_model.predict(img, conf=0.5)[0]

if len(results.boxes) == 0:
    print("❌ YOLO không tìm thấy biển số.")
else:
    for idx, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # 1. Padding nới rộng vùng cắt
        pad_h = int((y2 - y1) * 0.15)
        pad_w = int((x2 - x1) * 0.10)
        crop = img[max(0, y1 - pad_h):min(img.shape[0], y2 + pad_h),
        max(0, x1 - pad_w):min(img.shape[1], x2 + pad_w)]
        if crop.size == 0: continue

        # 2. Tiền xử lý nâng cao
        crop_res = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(crop_res, cv2.COLOR_BGR2GRAY)

        # Cân bằng sáng
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # --- BƯỚC QUAN TRỌNG: LÀM NÉT CẠNH (Giúp phân biệt 7/1) ---
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        cv2.imshow(f"AI nhin thay {idx}", sharpened)

        # 3. Nhận diện
        ocr_results = reader.readtext(sharpened, detail=1)

        # Sắp xếp theo tọa độ X trước (từ trái sang phải)
        # sau đó mới theo tọa độ Y (từ trên xuống dưới)
        ocr_results.sort(key=lambda x: (x[0][0][1] // 10, x[0][0][0]))

        plate_parts = []
        for (bbox, text, prob) in ocr_results:
            if prob > 0.2:
                # Chỉ lấy chữ và số, bỏ dấu gạch, dấu chấm
                clean_part = re.sub(r'[^0-9A-Z]', '', text.upper())
                plate_parts.append(clean_part)

        raw_plate = "".join(plate_parts)

        # 4. Hậu xử lý thông minh (Fix lỗi 1/7, 5/6 nhưng không làm mất chuỗi)
        final_text = raw_plate
        if len(final_text) >= 7:
            chars = list(final_text)
            # Fix lỗi số 1 và 7 phổ biến
            map_to_num = {'I': '1', 'T': '7', 'S': '5', 'G': '6', 'B': '8', 'D': '0'}
            # Thử fix các vị trí chắc chắn là số (thường là các vị trí cuối)
            for i in range(len(chars) - 1, len(chars) - 4, -1):
                if not chars[i].isdigit():
                    chars[i] = map_to_num.get(chars[i], chars[i])
            final_text = "".join(chars)

        print(f"✅ KẾT QUẢ CUỐI: {final_text}")

        # --- VẼ LÊN ẢNH ---
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y1 - 35), (x1 + 250, y1), (0, 255, 0), -1)
        cv2.putText(img, final_text, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

cv2.imshow("Anh dau vao", img)
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