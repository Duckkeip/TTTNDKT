import pandas as pd
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
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def init_db():
    # Ưu tiên lấy từ Secrets (Cloud) hoặc .env (Local)
    uri = os.getenv("MONGO_URI")

    # Nếu cả 2 đều không có (phòng hờ), mới dùng link cứng hoặc báo lỗi
    if not uri:
        st.error("Chưa cấu hình MONGO_URI trong Secrets hoặc .env!")
        st.stop()

    client = MongoClient(uri)
    return client["TN"]


db = init_db()

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
    # --- PHẦN LOG DỮ LIỆU THÔ ---
    # Log này giúp bạn debug xem OCR có đọc sót dòng nào không
    print("\n" + "="*30)
    print("DEBUG OCR RAW DATA:")
    for idx, text in enumerate(ocr_list):
        print(f"[{idx}]: {text}")
    print("="*30 + "\n")

    data = {
        "Họ và tên": "Không rõ",
        "Ngày sinh": "Không rõ",
        "Ngành": "Không rõ",
        "Khóa": "Không rõ",
        "Mã SV": "Không rõ",
        "Mã thẻ ngân hàng": "Không rõ",
        "Ngày hiệu lực / Hạn tới": "Không rõ"
    }

    # Chuyển list sang chữ in hoa, không dấu để dễ so khớp từ khóa
    def simple_clean(t):
        return "".join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn').upper()

    # Tiền xử lý list để so khớp từ khóa chính xác hơn
    # Loại bỏ khoảng trắng thừa và chuyển về chữ HOA
    clean_list = [str(line).strip().upper() for line in ocr_list if line]
    full_text = " | ".join(clean_list)

    # 1. Tìm Mã SV (Quan trọng nhất)
    mssv_match = re.search(r"\b\d{10}\b", full_text)
    if mssv_match:
        data["Mã SV"] = mssv_match.group(0)
        # Suy luận khóa từ MSSV (ví dụ: 23... -> 2023)
        if data["Khóa"] == "Không rõ":
            data["Khóa"] = f"20{data['Mã SV'][:2]}"

    # 2. Tìm Mã thẻ ngân hàng (16 số)
    card_match = re.search(r"9704\s?\d{4}\s?\d{4}\s?\d{4}", full_text)
    if card_match:
        data["Mã thẻ ngân hàng"] = card_match.group(0).replace(" ", "")

    # 3. Tìm các mốc thời gian
    dates = re.findall(r"\d{2}/\d{2}/\d{4}", full_text)
    if dates: data["Ngày sinh"] = dates[0]

    expiry = re.findall(r"\d{2}/\d{2}", full_text)
    if len(expiry) >= 2:
        data["Ngày hiệu lực / Hạn tới"] = f"{expiry[-2]} - {expiry[-1]}"

    # 4. Duyệt từng dòng để tìm các thông tin có nhãn (Label)
    for i, line in enumerate(ocr_list):
        # Tạo bản tạm không dấu để so khớp từ khóa cho chuẩn
        line_no_accent = "".join(
            c for c in unicodedata.normalize('NFD', line) if unicodedata.category(c) != 'Mn').upper()

        if any(k in line_no_accent.replace(" ", "") for k in ["HOVATEN", "TEN"]):
            # Dùng regex xóa sạch phần nhãn (Họ và tên, Hovà tén,...) để lấy tên
            name_part = re.sub(r'^.*?(H[OỌ].*?T[EÉ]N|TEN)[:\s]*', '', line, flags=re.IGNORECASE).strip()

            if len(name_part) > 5:
                data["Họ và tên"] = name_part.title()  # Hoặc .upper() tùy bạn
                break

        # 4. CHIẾN THUẬT DỰ PHÒNG (Nếu tên vẫn "Không rõ")
        # Trên thẻ VAA, tên luôn nằm TRÊN dòng "Ngày sinh"
    if data["Họ và tên"] == "Không rõ" and data["Ngày sinh"] != "Không rõ":
        for i, line in enumerate(clean_list):
            if data["Ngày sinh"] in line and i > 0:
                # Lấy dòng ngay phía trên dòng Ngày sinh làm tên
                potential_name = ocr_list[i - 1].strip()
                if len(potential_name) > 5 and not any(c.isdigit() for c in potential_name):
                    data["Họ và tên"] = potential_name.title()

        # 5. Tìm Ngành & Khóa (Ghi đè nếu thấy)
    for i, line in enumerate(clean_list):
        if "NGANH" in line:
            data["Ngành"] = ocr_list[i + 1].strip() if i + 1 < len(ocr_list) else "Không rõ"
        if "KHOA" in line or "KHOO" in line:  # Sửa lỗi OCR đọc nhầm Khóa thành Khoo
            year = re.search(r"20\d{2}", " ".join(ocr_list[i:i + 2]))
            if year: data["Khóa"] = year.group(0)

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


def get_student_from_db(student_id):
    """Tìm kiếm sinh viên linh hoạt (String/Int)"""
    clean_id = str(student_id).strip().replace('"', '')
    query = {
        "$or": [
            {"student_id": clean_id},
            {"student_id": int(clean_id) if clean_id.isdigit() else None}
        ]
    }
    return students_col.find_one(query)


def save_gate_event(plate, raw_info, image_bytes):
    """Ghi log hoặc Alert vào Database"""
    now = datetime.now()
    os.makedirs("images", exist_ok=True)
    img_name = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
    img_path = f"images/{img_name}"

    # Lưu ảnh vật lý (Dành cho chạy Local)
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    mssv_ocr = raw_info.get("Mã SV", "Không rõ")
    student_db = get_student_from_db(mssv_ocr)

    if not student_db:
        # Ghi Alert nếu không thấy MSSV
        alerts_col.insert_one({
            "time": now,
            "reason": "Student ID not registered",
            "student_ocr": raw_info,
            "plate_detected": plate,
            "image_path": img_path
        })
        return None, False

    # So khớp biển số
    def clean_p(p): return "".join(filter(str.isalnum, str(p))).upper()

    is_match = clean_p(plate) == clean_p(student_db.get("plate", ""))

    # Ghi Log thành công
    logs_col.insert_one({
        "time": now,
        "student_id": student_db["student_id"],
        "student_name": student_db["full_name"],
        "plate_detected": plate,
        "image_path": img_path,
        "status": "IN",
        "note": "Match plate" if is_match else "Plate mismatch"
    })
    return student_db, is_match
def get_student_from_db(student_id):
    """Tìm kiếm sinh viên linh hoạt (String/Int)"""
    clean_id = str(student_id).strip().replace('"', '')
    query = {
        "$or": [
            {"student_id": clean_id},
            {"student_id": int(clean_id) if clean_id.isdigit() else None}
        ]
    }
    return students_col.find_one(query)


def save_gate_event(plate, raw_info, image_bytes):
    """Ghi log hoặc Alert vào Database"""
    now = datetime.now()
    os.makedirs("images", exist_ok=True)
    img_name = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
    img_path = f"images/{img_name}"

    # Lưu ảnh vật lý (Dành cho chạy Local)
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    mssv_ocr = raw_info.get("Mã SV", "Không rõ")
    student_db = get_student_from_db(mssv_ocr)

    if not student_db:
        # Ghi Alert nếu không thấy MSSV
        alerts_col.insert_one({
            "time": now,
            "reason": "Student ID not registered",
            "student_ocr": raw_info,
            "plate_detected": plate,
            "image_path": img_path
        })
        return None, False

    # So khớp biển số
    def clean_p(p): return "".join(filter(str.isalnum, str(p))).upper()

    is_match = clean_p(plate) == clean_p(student_db.get("plate", ""))

    # Ghi Log thành công
    logs_col.insert_one({
        "time": now,
        "student_id": student_db["student_id"],
        "student_name": student_db["full_name"],
        "plate_detected": plate,
        "image_path": img_path,
        "status": "IN",
        "note": "Match plate" if is_match else "Plate mismatch"
    })
    return student_db, is_match

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
            # SỬA LỖI Ở ĐÂY: Lấy đúng key "enhanced"
            res_plate = advanced_enhance(crop)
            ocr_res = reader.readtext(res_plate["enhanced"], detail=0)

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

        # Padding mở rộng vùng cắt
        h_img, w_img = img.shape[:2]
        pad = 15
        crop = img[max(0, y1 - pad):min(h_img, y2 + pad), max(0, x1 - pad):min(w_img, x2 + pad)]

        if cls_name == "the" and crop.size > 0:
            mssv = raw_info["Mã SV"]

            if mssv != "Không rõ":
                # Gọi hàm tìm kiếm trực tiếp thay vì requests.get
                student_db = get_student_from_db(mssv)

                if student_db:
                    st.success(f"✅ Tìm thấy: {student_db['full_name']}")
                    # Cập nhật thông tin chuẩn từ DB vào final_info để hiển thị
                    raw_info["Họ và tên"] = student_db["full_name"]
                    raw_info["Ngành"] = student_db.get("major", "N/A")

                    # Ghi log sự kiện (Thay thế cho @app.post /api/gate-event)
                    # Giả sử 'plate_val' là biển số bạn đã nhận diện được trước đó
                    save_gate_event(plate_val, raw_info, img_encoded_bytes)
                else:
                    st.error(f"❌ MSSV {mssv} không tồn tại trên hệ thống!")
            res = advanced_enhance(crop)

            # --- HIỂN THỊ ẢNH ĐANG XỬ LÝ LÊN APP ĐỂ CHECK ---

            with st.expander(" Chi tiết xử lý vùng thẻ (Debug)"):col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.image(res["raw_resized"], caption="Ảnh Gốc (Resized)")
            col_c2.image(res["enhanced"], caption="Ảnh Enhanced (CLAHE)")
            # Nếu bạn muốn xem ảnh mờ hay không, nhìn vào đây là rõ nhất

            # 1. OCR đọc chữ từ ảnh
            ocr_list = reader.readtext(res["enhanced"], detail=0)

            # --- HIỂN THỊ NHẬT KÝ QUÉT CHỮ ---
            with st.expander(" Nhật ký quét chữ (OCR Log)", expanded=False):
                st.write("Dữ liệu thô AI đọc được từ ảnh:")
                st.code(ocr_list)

            # 2. Trích xuất thông tin thô (Lúc này có thể sai dấu/thiếu thông tin)
            raw_info = extract_student_info(ocr_list)

            # Hiển thị dữ liệu thô vừa trích xuất được

            with st.expander(" Chi tiết dữ liệu OCR trích xuất", expanded=True):
                # Tạo bảng từ dictionary
                df_raw = pd.DataFrame(list(raw_info.items()), columns=["Trường thông tin", "Giá trị đọc được"])
                st.table(df_raw)

            # 3. --- LOGIC ĐỐI CHIẾU QUA API SERVER ---
            final_info = raw_info.copy()

            if raw_info["Mã SV"] != "Không rõ":
                try:
                    # Gọi API Server để lấy dữ liệu chuẩn từ MongoDB Atlas
                    # Đảm bảo api_server.py đang chạy ở port 8000
                    response = requests.get(f"http://127.0.0.1:8000/api/student/{raw_info['Mã SV']}", timeout=5)

                    if response.status_code == 200:
                        student_db = response.json()

                        # --- HIỂN THỊ BẢNG ĐỐI CHIẾU ---
                        st.markdown("### 📊 Log đối chiếu: OCR vs Database")
                        with st.container():
                            c1, c2 = st.columns(2)
                            with c1:
                                st.info("📝 **Kết quả OCR (Thô)**")
                                st.write(f"- Họ tên: `{raw_info['Họ và tên']}`")
                                st.write(f"- MSSV: `{raw_info['Mã SV']}`")
                                st.write(f"- Ngày sinh: `{raw_info['Ngày sinh']}`")

                            with c2:
                                st.success("✅ **Database (Chuẩn)**")
                                st.write(f"- Họ tên: **{student_db.get('full_name')}**")
                                st.write(f"- MSSV: **{student_db.get('student_id')}**")
                                st.write(f"- Ngày sinh: **{student_db.get('birthday')}**")

                        # GHI ĐÈ DỮ LIỆU CHUẨN ĐỂ HIỂN THỊ & GỬI API
                        final_info["Họ và tên"] = student_db.get("full_name", raw_info["Họ và tên"])
                        final_info["Ngành"] = student_db.get("major", raw_info["Ngành"])
                        final_info["Khóa"] = student_db.get("batch", raw_info["Khóa"])
                        final_info["Mã thẻ ngân hàng"] = student_db.get("bank_card", raw_info["Mã thẻ ngân hàng"])
                        final_info["Ngày hiệu lực / Hạn tới"] = student_db.get("expiry_date",
                                                                               raw_info["Ngày hiệu lực / Hạn tới"])

                    else:
                        st.error(f"❌ Server báo: Không tìm thấy MSSV {raw_info['Mã SV']} trong Database!")

                except Exception as e:
                    st.warning(f"⚠️ Không thể kết nối tới API Server để đối chiếu. Lỗi: {e}")

            # Lưu vào danh sách kết quả cuối cùng
            results_data["students"].append(final_info)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 3. GỬI API (Sử dụng dữ liệu đã được Database sửa lỗi)
    if results_data["plates"] or results_data["students"]:
        plate = results_data["plates"][0] if results_data["plates"] else "unknown"
        student = results_data["students"][0] if results_data["students"] else None
        send_to_api(img, plate, student)

    return display_img, results_data

# ==========================================
# 4. GIAO DIỆN STREAMLIT
# ==========================================

st.title("VAA Hệ thống giữ xe thẻ sinh viên")
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

