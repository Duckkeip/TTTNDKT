import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from tkinter import Tk, filedialog
from datetime import datetime
import unicodedata
# =========================
# QUALITY CHECK & ENHANCE
# =========================

def is_blurry(image, thresh=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < thresh, fm

def is_dark(image, thresh=60):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return mean < thresh, mean

def remove_shadow(img):
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        result_planes.append(diff)

    result = cv2.merge(result_planes)
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8,8))
    cl1 = clahe.apply(gray)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(cl1, -1, kernel)
    return sharp


# =========================
# 1. KHỞI TẠO MODEL
# =========================
yolo_plate = YOLO("Bienso.pt")
yolo_sv    = YOLO("Thesv.pt")
reader = easyocr.Reader(['vi','en'], gpu=False)

# =========================
# 2. CHỌN ẢNH
# =========================
def select_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title="Chọn ảnh")
    root.destroy()
    return path

# =========================
# 3. SỬA LỖI BIỂN SỐ
# =========================
def vietnamese_plate_correction(text):
    text = re.sub(r'[^0-9A-Z]', '', text.upper())
    if len(text) < 7:
        return text

    chars = list(text)

    map_to_char = {'1': 'I', '7': 'T', '0': 'O', '5': 'S', '2': 'Z'}
    map_to_num  = {'I': '1', 'T': '7', 'S': '5', 'G': '6', 'B': '8', 'D': '0', 'O':'0'}

    if len(chars) > 2 and chars[2].isdigit():
        chars[2] = map_to_char.get(chars[2], chars[2])

    if len(chars) > 3 and not chars[3].isdigit():
        chars[3] = map_to_num.get(chars[3], chars[3])

    for i in range(len(chars)-1, max(len(chars)-4, 3), -1):
        if not chars[i].isdigit():
            chars[i] = map_to_num.get(chars[i], chars[i])

    return "".join(chars)

# =========================
# 4. MAIN
# =========================
image_path = select_file()
if not image_path:
    print("❌ Chưa chọn ảnh")
    exit()

img = cv2.imread(image_path)

plate_results = yolo_plate.predict(img, conf=0.5)[0]
sv_results    = yolo_sv.predict(img, conf=0.5)[0]
if len(sv_results.boxes) == 0:
    print("⚠ KHÔNG PHÁT HIỆN THẺ SINH VIÊN HOẶC KHUÔN MẶT")

if len(plate_results.boxes) == 0:
    print("⚠ KHÔNG PHÁT HIỆN BIỂN SỐ")
os.makedirs("plates", exist_ok=True)
os.makedirs("sv_cards", exist_ok=True)

def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.upper()
    text = re.sub(r"[^A-Z0-9/ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def extract_student_info(ocr_text):
    data = {
        "school": "",
        "name": "",
        "student_id": "",
        "major": "",
        "year": "",
        "dob": "",  # ✅ ngày sinh
        "dates": []
    }

    text = normalize_text(ocr_text)
    # 🎂 DATE OF BIRTH / NGÀY SINH (ưu tiên có nhãn)
    dob_match = re.search(
        r"(NGAY SINH|DOB|DATE OF BIRTH)\s*[: ]*\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        text
    )

    if dob_match:
        data["dob"] = dob_match.group(2)
    else:
        # fallback: lấy ngày dạng dd/mm/yyyy
        full_dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
        if full_dates:
            data["dob"] = full_dates[0]

    # 🎓 Trường
    if "HOC VIEN HANG KHONG" in text or "HỌC VIỆN HÀNG KHÔNG" in text:
        data["school"] = "HỌC VIỆN HÀNG KHÔNG VIỆT NAM"

    # 🆔 MSSV (>=8 số)
    ids = re.findall(r"\b\d{8,11}\b", text)
    if ids:
        data["student_id"] = ids[0]

    # 📅 Năm
    years = re.findall(r"\b20\d{2}\b", text)
    if years:
        data["year"] = years[0]

    # 📆 Ngày
    data["dates"] = re.findall(r"\b\d{1,2}/\d{1,2}\b", text)

    # 📚 NGÀNH (bắt theo từ khóa)
    if ("CONG" in text and "NGHE" in text and "THONG" in text) or \
       ("NGHE" in text and "TIN" in text):
        data["major"] = "CÔNG NGHỆ THÔNG TIN"
    elif "HANG KHONG" in text:
        data["major"] = "HÀNG KHÔNG"

    # 👤 HỌ TÊN (chịu lỗi OCR)
    name_patterns = [
        r"HO.?V.?TEN[:\s]+([A-Z\s]+)",
        r"HO.?VA.?TEN[:\s]+([A-Z\s]+)",
        r"HOVATEN[:\s]+([A-Z\s]+)"
    ]

    for p in name_patterns:
        m = re.search(p, text)
        if m:
            name = m.group(1)
            name = re.sub(r"(MSSV|MA SV|NGANH|KHOA|SINH VIEN|THE).*", "", name)
            name = re.sub(r"\s+", " ", name).strip()
            data["name"] = name.title()
            break

    return data

# =========================
# 5. NHẬN DIỆN BIỂN SỐ
# =========================
print("\n🚗 ===== NHẬN DIỆN BIỂN SỐ =====")

for idx, box in enumerate(plate_results.boxes):

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    yolo_conf = float(box.conf[0])

    pad_h = int((y2 - y1) * 0.15)
    pad_w = int((x2 - x1) * 0.10)

    crop = img[max(0, y1-pad_h):min(img.shape[0], y2+pad_h),
               max(0, x1-pad_w):min(img.shape[1], x2+pad_w)]

    if crop.size == 0:
        continue

    crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(2.0, (8,8))
    enhanced = clahe.apply(gray)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    cv2.imwrite(f"plates/plate_{idx}.jpg", sharpened)

    ocr_results = reader.readtext(sharpened, detail=1)
    ocr_results.sort(key=lambda x: (x[0][0][1]//10, x[0][0][0]))

    plate_parts = []
    conf_list = []

    for (bbox, text, prob) in ocr_results:
        if prob > 0.2:
            clean = re.sub(r'[^0-9A-Z]', '', text.upper())
            if clean:
                plate_parts.append(clean)
                conf_list.append(prob)

    raw_plate = "".join(plate_parts)
    ocr_conf = np.mean(conf_list) if conf_list else 0

    fixed_plate = vietnamese_plate_correction(raw_plate)
    final_conf = round((yolo_conf*0.5 + ocr_conf*0.5)*100, 2)

    is_valid = bool(re.match(r'^\d{2}[A-Z]\d{4,5}$', fixed_plate))

    if final_conf < 35 or len(fixed_plate) < 7:
        continue

    print("────────────────────────")
    print(f"📌 Plate : {fixed_plate}")
    print(f"🎯 FINAL : {final_conf}%")
    print(f"🇻🇳 Check: {'HỢP LỆ' if is_valid else 'NGHI NGỜ'}")

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.rectangle(img, (x1, y1-40), (x1+380, y1), (0,255,0), -1)
    cv2.putText(img, f"{fixed_plate} | {final_conf}%", (x1+5, y1-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

# =========================
# 6. NHẬN DIỆN THẺ SINH VIÊN
# =========================
print("\n🎓 ===== NHẬN DIỆN THẺ SINH VIÊN =====")

names = yolo_sv.names  # {0:'the', 1:'mat'}

for idx, box in enumerate(sv_results.boxes):

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    cls_name = names[cls_id]

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # ================= FACE =================
    if cls_name == "mat":
        os.makedirs("faces", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"faces/face_{timestamp}_{idx}.jpg", crop)

        print("────────────────────────")
        print(f"🙂 Khuôn mặt #{idx}")
        print(f"📦 YOLO : {conf:.2f}")

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(img, "KHUON MAT", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ================= STUDENT CARD =================
    elif cls_name == "the":
        os.makedirs("sv_cards", exist_ok=True)
        cv2.imwrite(f"sv_cards/card_{idx}.jpg", crop)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(gray,255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,11,2)

        ocr_sv = reader.readtext(thresh, detail=1)

        texts = []
        for (bbox, text, prob) in ocr_sv:
            if prob > 0.3:
                texts.append(text)

        full_text = " | ".join(texts)
        info = extract_student_info(full_text)

        print("🎯 TRÍCH THÔNG TIN:")
        for k, v in info.items():
            print(f"   {k.upper():10}: {v}")
        print("────────────────────────")
        print(f"🎓 Thẻ SV #{idx}")
        print(f"📦 YOLO : {conf:.2f}")
        print(f"🪪 OCR  : {full_text}")

        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.rectangle(img, (x1, y1-30), (x1+300, y1), (255,0,0), -1)
        label = f"{info['student_id']} - {info['name'][:20]}"
        cv2.rectangle(img, (x1, y1 - 35), (x1 + 480, y1), (255, 0, 0), -1)
        cv2.putText(img, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# =========================
# 7. HIỂN THỊ KẾT QUẢ
# =========================
cv2.imshow("KET QUA", img)
cv2.imwrite("result.jpg", img)
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