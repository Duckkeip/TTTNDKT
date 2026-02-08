import streamlit as st
import cv2
import numpy as np
import re
import os
import unicodedata
from ultralytics import YOLO
import easyocr
import requests
import base64
from datetime import datetime


def send_to_api(frame, plate, student_info):
    """
    Gửi dữ liệu nhận diện về Server.
    Đã tối ưu hóa để tương thích với Template sinh viên mới.
    """
    # 1. Lấy thời gian hiện tại từ máy khách (client)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Xử lý ảnh trước khi gửi
    # Nếu frame quá lớn, nên resize nhẹ để giảm tải băng thông
    h, w = frame.shape[:2]
    if w > 1000:
        new_w = 800
        new_h = int(h * (new_w / w))
        frame = cv2.resize(frame, (new_w, new_h))

    # Nén chất lượng JPEG xuống 70-80% để cân bằng giữa độ nét và tốc độ
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    success, buffer = cv2.imencode('.jpg', frame, encode_param)

    if not success:
        st.error("❌ Lỗi mã hóa ảnh!")
        return

    img_base64 = base64.b64encode(buffer).decode()

    # 3. Chuẩn bị Payload
    # Đảm bảo student_info là dictionary chuẩn từ hàm extract_student_info
    payload = {
        "plate": plate,
        "student": student_info,  # Gồm Họ tên, MSSV, Ngành, Khóa, Số thẻ, Hạn dùng
        "image": img_base64,
        "client_time": current_time
    }

    # 4. Gửi request (Sử dụng khối try-except để không làm sập App Streamlit)
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/gate-event",
            json=payload,
            timeout=3  # Timeout ngắn để tránh chờ đợi lâu trong luồng camera
        )

        if response.status_code == 200:
            st.toast(f"✅ Đã gửi dữ liệu: {plate}", icon="📡")
        else:
            # Ghi log lỗi từ Server phản hồi
            st.error(f"❌ Server lỗi ({response.status_code}): {response.text}")

    except requests.exceptions.Timeout:
        st.warning("⚠️ API Server phản hồi quá chậm (Timeout)!")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Không thể kết nối tới API Server (Check http://127.0.0.1:8000)")
    except Exception as e:
        st.error(f"⚠️ Lỗi kết nối: {str(e)}")
# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO (Dùng Cache để chạy nhanh)
# ==========================================
st.set_page_config(page_title="Hệ thống AI Giữ xe VAA", layout="wide")


@st.cache_resource
def load_models():
    # Sử dụng os.path.join để đường dẫn chạy được trên cả Windows/Linux/Mac
    base_path = os.getcwd()  # Lấy thư mục hiện tại của dự án

    plate_path = os.path.join(base_path, "models", "Bienso.pt")
    sv_path = os.path.join(base_path, "models", "Thesv.pt")

    # Kiểm tra file có tồn tại không trước khi load
    if not os.path.exists(plate_path) or not os.path.exists(sv_path):
        st.error(f"Không tìm thấy file model tại: {os.path.dirname(plate_path)}")
        st.stop()

    yolo_plate = YOLO(plate_path)
    yolo_sv = YOLO(sv_path)
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


def extract_student_info(ocr_list):
    data = {
        "Họ và tên": "Không rõ",
        "Ngày sinh": "Không rõ",
        "Ngành": "Không rõ",
        "Khóa": "Không rõ",
        "Mã SV": "Không rõ",
        "Mã thẻ ngân hàng": "Không rõ",
        "Ngày hiệu lực / Hạn tới": "Không rõ"
    }

    # Chuyển list thành chuỗi lớn để tìm các định dạng số cố định
    full_text = " | ".join(ocr_list).upper()

    # 1. Tìm Mã SV (10 số)
    mssv_match = re.search(r"\b\d{10}\b", full_text)
    if mssv_match: data["Mã SV"] = mssv_match.group(0)

    # 2. Tìm Mã thẻ ngân hàng (16 số, bắt đầu bằng 9704)
    card_match = re.search(r"9704\s?\d{4}\s?\d{4}\s?\d{4}", full_text)
    if card_match: data["Mã thẻ ngân hàng"] = card_match.group(0).replace(" ", "")

    # 3. Tìm các mốc thời gian (dd/mm/yyyy và mm/yy)
    dates = re.findall(r"\d{2}/\d{2}/\d{4}", full_text)
    if dates: data["Ngày sinh"] = dates[0]

    expiry = re.findall(r"\d{2}/\d{2}", full_text)
    if len(expiry) >= 2:
        data["Ngày hiệu lực / Hạn tới"] = f"{expiry[-2]} - {expiry[-1]}"

    # 4. Duyệt từng phần tử để tìm Họ tên, Khóa, Ngành
    for i, line in enumerate(ocr_list):
        line_clean = line.strip().upper()

        # Tìm Họ tên (Dựa vào vị trí dòng)
        if any(k in line_clean for k in ["HO VA TEN", "HOVATEN", "TEN"]):
            if ":" in line:
                data["Họ và tên"] = line.split(":")[-1].strip().title()
            elif i + 1 < len(ocr_list):
                data["Họ và tên"] = ocr_list[i + 1].strip().title()

        # Tìm Khóa (Tìm số 4 chữ số nằm gần chữ KHOA)
        if "KHOA" in line_clean:
            # Bước 1: Tìm ngay trong dòng đó xem có số 4 chữ số không (ví dụ: KHOA: 2023)
            year_match = re.search(r"20\d{2}", line)  # Tìm năm bắt đầu bằng 20xx
            if year_match:
                data["Khóa"] = year_match.group(0)

            # Bước 2: Nếu dòng đó không có, tìm ở 2 dòng lân cận (phòng trường hợp OCR nhảy dòng)
            else:
                context = ""
                if i > 0: context += ocr_list[i - 1]
                if i + 1 < len(ocr_list): context += ocr_list[i + 1]

                year_match_context = re.search(r"20\d{2}", context)
                if year_match_context:
                    data["Khóa"] = year_match_context.group(0)

        # Tìm Ngành
        if "NGANH" in line_clean:
            if ":" in line:
                data["Ngành"] = line.split(":")[-1].strip()
            elif i + 1 < len(ocr_list):
                data["Ngành"] = ocr_list[i + 1].strip()

    return data


def advanced_enhance(image):
    if image is None or image.size == 0:
        return None

    # 1. Phóng to là quan trọng nhất (Dùng Lanczos4)
    h, w = image.shape[:2]
    resized = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # 2. Tạo bản Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE nhẹ (Giảm clipLimit xuống 1.2 để không bị cháy ảnh)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return {
        "enhanced": enhanced,
        "raw_resized": resized
    }
# ==========================================
# 3. HÀM XỬ LÝ CHÍNH (DEEP SCAN)
# ==========================================

def process_frame(img):
    display_img = img.copy()
    results_data = {"plates": [], "students": []}

    # --- 1. XỬ LÝ BIỂN SỐ ---
    plate_results = yolo_plate.predict(img, conf=0.5, verbose=False)[0]
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            # SỬ DỤNG HÀM ENHANCE MỚI
            enhanced_plate = advanced_enhance(crop)
            ocr_res = reader.readtext(enhanced_plate, detail=0)

            raw_plate = "".join(ocr_res).upper()
            fixed_plate = vietnamese_plate_correction(raw_plate)

            results_data["plates"].append(fixed_plate)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_img, fixed_plate, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- 2. XỬ LÝ THẺ SINH VIÊN ---
    sv_results = yolo_sv.predict(img, conf=0.5, verbose=False)[0]
    for box in sv_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = yolo_sv.names[int(box.cls[0])]

        # Mở rộng vùng cắt một chút (Padding) để không mất mép chữ
        h_img, w_img = img.shape[:2]
        pad = 10
        crop = img[max(0, y1 - pad):min(h_img, y2 + pad), max(0, x1 - pad):min(w_img, x2 + pad)]

        if cls_name == "the" and crop.size > 0:
            res = advanced_enhance(crop)

            # Lần 1: Đọc trên ảnh đã Enhanced (Tốt cho MSSV, Số thẻ)
            ocr_list = reader.readtext(res["enhanced"], detail=0)
            info = extract_student_info(ocr_list)

            # KIỂM TRA: Nếu thiếu Họ tên hoặc Khóa, quét lại Lần 2 trên ảnh Raw Resized
            if info["Họ và tên"] == "Không rõ" or info["Khóa"] == "Không rõ":
                ocr_list_backup = reader.readtext(res["raw_resized"], detail=0)
                info_backup = extract_student_info(ocr_list_backup)

                # Cập nhật những gì bản cũ thiếu
                if info["Họ và tên"] == "Không rõ": info["Họ và tên"] = info_backup["Họ và tên"]
                if info["Khóa"] == "Không rõ": info["Khóa"] = info_backup["Khóa"]

            results_data["students"].append(info)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 3. GỬI API
    if results_data["plates"] or results_data["students"]:
        plate = results_data["plates"][0] if results_data["plates"] else "unknown"
        student = results_data["students"][0] if results_data["students"] else None
        send_to_api(img, plate, student)

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

