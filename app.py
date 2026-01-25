import streamlit as st
import cv2
import numpy as np
import re
import os
import unicodedata
from ultralytics import YOLO
import easyocr
from datetime import datetime

# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO (Dùng Cache để chạy nhanh)
# ==========================================
st.set_page_config(page_title="Hệ thống AI Giữ xe VAA", layout="wide")


@st.cache_resource
def load_models():
    # Load YOLO
    yolo_plate = YOLO("models/Bienso.pt")
    yolo_sv = YOLO("models/Thesv.pt")
    # Load EasyOCR
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    return yolo_plate, yolo_sv, reader


yolo_plate, yolo_sv, reader = load_models()


# ==========================================
# 2. CÁC HÀM LOGIC CŨ CỦA BẠN (ĐÃ TỐI ƯU)
# ==========================================

def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.upper()
    text = re.sub(r"[^A-Z0-9/ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def vietnamese_plate_correction(text):
    text = re.sub(r'[^0-9A-Z]', '', text.upper())
    if len(text) < 7: return text
    chars = list(text)
    map_to_char = {'1': 'I', '7': 'T', '0': 'O', '5': 'S', '2': 'Z'}
    map_to_num = {'I': '1', 'T': '7', 'S': '5', 'G': '6', 'B': '8', 'D': '0', 'O': '0'}
    if len(chars) > 2 and chars[2].isdigit():
        chars[2] = map_to_char.get(chars[2], chars[2])
    if len(chars) > 3 and not chars[3].isdigit():
        chars[3] = map_to_num.get(chars[3], chars[3])
    for i in range(len(chars) - 1, max(len(chars) - 4, 3), -1):
        if not chars[i].isdigit():
            chars[i] = map_to_num.get(chars[i], chars[i])
    return "".join(chars)


def extract_student_info(ocr_text):
    data = {"Họ Tên": "Không rõ", "MSSV": "Không rõ", "Ngành": "Không rõ", "Ngày sinh": "Không rõ"}
    text = normalize_text(ocr_text)

    # Ngày sinh
    dob_match = re.search(r"(NGAY SINH|DOB|DATE OF BIRTH)\s*[: ]*\s*(\d{1,2}/\d{1,2}/\d{2,4})", text)
    if dob_match: data["Ngày sinh"] = dob_match.group(2)

    # MSSV
    ids = re.findall(r"\b\d{8,11}\b", text)
    if ids: data["MSSV"] = ids[0]

    # Ngành
    if any(x in text for x in ["CONG NGHE", "TIN HOC", "CNTT"]):
        data["Ngành"] = "CÔNG NGHỆ THÔNG TIN"
    elif "HANG KHONG" in text:
        data["Ngành"] = "HÀNG KHÔNG"

    # Họ tên
    name_patterns = [r"HO.?V.?TEN[:\s]+([A-Z\s]+)", r"HOVATEN[:\s]+([A-Z\s]+)"]
    for p in name_patterns:
        m = re.search(p, text)
        if m:
            name = re.sub(r"(MSSV|MA SV|NGANH|KHOA|SINH VIEN|THE).*", "", m.group(1))
            data["Họ Tên"] = name.strip().title()
            break
    return data


# ==========================================
# 3. HÀM XỬ LÝ CHÍNH (DEEP SCAN)
# ==========================================

def process_frame(img):
    display_img = img.copy()
    results_data = {"plates": [], "students": []}

    # --- Xử lý Biển số ---
    plate_results = yolo_plate.predict(img, conf=0.5, verbose=False)[0]
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            # Tiền xử lý giống code cũ của bạn
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            enhanced = clahe.apply(gray)
            ocr_res = reader.readtext(enhanced, detail=0)
            raw_plate = "".join(ocr_res).upper()
            fixed_plate = vietnamese_plate_correction(raw_plate)

            results_data["plates"].append(fixed_plate)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_img, fixed_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- Xử lý Thẻ sinh viên ---
    sv_results = yolo_sv.predict(img, conf=0.5, verbose=False)[0]
    for box in sv_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = yolo_sv.names[int(box.cls[0])]
        crop = img[y1:y2, x1:x2]

        if cls_name == "the" and crop.size > 0:
            ocr_sv = reader.readtext(crop, detail=0)
            info = extract_student_info(" | ".join(ocr_sv))
            results_data["students"].append(info)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return display_img, results_data


# ==========================================
# 4. GIAO DIỆN STREAMLIT
# ==========================================

st.title("🛡️ VAA Gate Control System")
source = st.sidebar.radio("Nguồn đầu vào", ["📷 Camera", "📁 Tải ảnh lên"])

if source == "📁 Tải ảnh lên":
    file = st.file_uploader("Chọn ảnh thẻ SV hoặc Biển số", type=['jpg', 'png', 'jpeg'])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        res_img, data = process_frame(img)

        col1, col2 = st.columns(2)
        col1.image(img, channels="BGR", caption="Ảnh gốc")
        col2.image(res_img, channels="BGR", caption="Ảnh nhận diện")

        if data["plates"]: st.success(f"Biển số tìm thấy: {', '.join(data['plates'])}")
        if data["students"]:
            st.write("### Thông tin sinh viên:")
            st.table(data["students"])

else:
    col_vid, col_res = st.columns([2, 1])
    with col_vid:
        run = st.checkbox("Bật Camera")
        capture = st.button("📸 CHỤP & QUÉT")
        window = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret: break

            # Hiển thị luồng trực tiếp
            window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if capture:
                with col_res:
                    st.info("Đang phân tích...")
                    res_img, data = process_frame(frame)
                    st.image(res_img, channels="BGR")
                    if data["plates"]: st.success(f"Biển số: {data['plates'][0]}")
                    if data["students"]: st.table(data["students"])
                capture = False
        cap.release()