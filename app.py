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

# Khởi tạo bộ nhớ tạm để "ghép cặp" nếu upload nhiều ảnh khác nhau
if 'pair_data' not in st.session_state:
    st.session_state.pair_data = {"mssv": None, "plate": None, "raw_info": None}
@st.cache_resource
def init_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        st.error("Chưa cấu hình MONGO_URI trong Secrets hoặc .env!")
        st.stop()

    client = MongoClient(uri)
    db = client["TN"]  # Tên database của bạn

    # Trả về các collection để dùng ở ngoài
    return db["students"], db["gate_logs"], db["alerts"]


# QUAN TRỌNG: Gán biến ở đây để các hàm khác như get_student_from_db có thể dùng được
students_col, logs_col, alerts_col = init_db()

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
        key = line_no_accent.replace(" ", "")
        if re.search(r'HOVATEN', key):
            name_part = re.sub(
                r'H[OỌ].*?V[ÀA].*?T[EÉ]N[:\s]*',
                '',
                line,
                flags=re.IGNORECASE
            ).strip()

            if len(name_part) > 5:
                data["Họ và tên"] = name_part.title()
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
    """Logic tìm kiếm linh hoạt nhất để tránh lỗi kiểu dữ liệu khi Public Cloud"""
    if not student_id: return None

    clean_id = str(student_id).strip().replace('"', '')
    # Thử tìm kiếm 3 trường hợp: Chữ chuẩn, Chữ có ngoặc, và Số nguyên
    query = {
        "$or": [
            {"student_id": clean_id},
            {"student_id": f'"{clean_id}"'},
            {"student_id": int(clean_id) if clean_id.isdigit() else None}
        ]
    }
    return students_col.find_one(query)


def check_gate_process(plate_detected, mssv_ocr):
    """
    Logic: Không cần đăng ký biển trước.
    Chỉ so khớp biển số lúc VÀO và lúc RA của cùng 1 thẻ SV.
    """
    now = datetime.now()

    # 1. Tìm sinh viên trong DB (Để biết thẻ này có hợp lệ không)
    student_db = students_col.find_one({"student_id": mssv_ocr})
    if not student_db:
        return "ERROR", f"Thẻ SV {mssv_ocr} không hợp lệ hoặc chưa kích hoạt!"

    # 2. Tìm lượt VÀO (IN) gần nhất của thẻ này mà CHƯA có lượt RA (OUT)
    last_entry = logs_col.find_one(
        {"student_id": mssv_ocr, "status": "IN"},
        sort=[("time", -1)]
    )

    def clean(p):
        return "".join(filter(str.isalnum, str(p))).upper()

    # --- TRƯỜNG HỢP: XE ĐANG RA ---
    if last_entry:
        plate_at_in = last_entry.get("plate_detected")

        # So khớp biển số lúc này với biển số lúc vào bãi
        if clean(plate_detected) == clean(plate_at_in):
            # KHỚP -> Cho ra
            logs_col.insert_one({
                "time": now,
                "student_id": mssv_ocr,
                "student_name": student_db["full_name"],
                "plate_detected": plate_detected,
                "status": "OUT",
                "note": "Ra bãi thành công (Khớp biển vào)"
            })
            return "SUCCESS_OUT", f"MỜI RA! Xe khớp với lúc vào ({plate_at_in})"
        else:
            # KHÔNG KHỚP -> Cảnh báo
            return "ALERT_THEFT", f"⚠️ SAI BIỂN SỐ! Lúc vào đi xe {plate_at_in}, lúc ra lại dắt xe {plate_detected}!"

    # --- TRƯỜNG HỢP: XE ĐANG VÀO ---
    else:
        logs_col.insert_one({
            "time": now,
            "student_id": mssv_ocr,
            "student_name": student_db["full_name"],
            "plate_detected": plate_detected,
            "status": "IN",
            "note": "Vào bãi"
        })
        return "SUCCESS_IN", f"MỜI VÀO! Đã ghi nhận xe {plate_detected} cho SV {student_db['full_name']}"
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

    # --- 1. NHẬN DIỆN BIỂN SỐ ---
    plate_results = yolo_plate.predict(img, conf=0.5, verbose=False)[0]
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            res_plate = advanced_enhance(crop)
            ocr_res = reader.readtext(res_plate["enhanced"], detail=0)
            raw_plate = "".join(ocr_res).upper()
            fixed_plate = vietnamese_plate_correction(raw_plate)

            results_data["plates"].append(fixed_plate)
            # Vẽ khung xanh lá cho biển số
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_img, fixed_plate, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- 2. NHẬN DIỆN THẺ SINH VIÊN ---
    sv_results = yolo_sv.predict(img, conf=0.5, verbose=False)[0]
    for box in sv_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = yolo_sv.names[int(box.cls[0])]

        h_img, w_img = img.shape[:2]
        pad = 15
        crop = img[max(0, y1 - pad):min(h_img, y2 + pad), max(0, x1 - pad):min(w_img, x2 + pad)]

        if cls_name == "the" and crop.size > 0:
            res = advanced_enhance(crop)

            # --- HIỂN THỊ DEBUG (Giữ nguyên theo ý bạn) ---
            with st.expander("🔍 Chi tiết xử lý vùng thẻ (Debug)"):
                col_c1, col_c2 = st.columns(2)
                col_c1.image(res["raw_resized"], caption="Ảnh Gốc")
                col_c2.image(res["enhanced"], caption="Ảnh Enhanced")

            ocr_list = reader.readtext(res["enhanced"], detail=0)
            with st.expander("📝 Nhật ký quét chữ (OCR Log)", expanded=False):
                st.code(ocr_list)

            raw_info = extract_student_info(ocr_list)

            with st.expander("📊 Chi tiết dữ liệu OCR trích xuất", expanded=True):
                df_raw = pd.DataFrame(list(raw_info.items()), columns=["Trường thông tin", "Giá trị đọc được"])
                st.table(df_raw)

            # Chỉ thêm vào danh sách nếu quét được Mã SV hợp lệ
            if raw_info["Mã SV"] != "Không rõ":
                results_data["students"].append(raw_info)

            # Vẽ khung xanh dương cho thẻ
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # --- 3. LOGIC KẾT HỢP (PAIRING): CHỈ XỬ LÝ KHI CÓ ĐỦ 2 ĐIỀU KIỆN ---
    if results_data["students"] and results_data["plates"]:
        # Lấy dữ liệu đầu tiên tìm thấy
        main_student = results_data["students"][0]
        main_plate = results_data["plates"][0]
        mssv = main_student["Mã SV"]

        # A. Lấy thông tin chuẩn từ Database
        student_db = get_student_from_db(mssv)

        if student_db:
            # --- HIỂN THỊ BẢNG ĐỐI CHIẾU ---
            st.markdown("### 📊 Log đối chiếu: OCR vs Database")
            with st.container():
                c1, c2 = st.columns(2)
                with c1:
                    st.info("📝 **Kết quả OCR (Thô)**")
                    st.write(f"- Họ tên: `{main_student['Họ và tên']}`")
                    st.write(f"- MSSV: `{mssv}`")
                with c2:
                    st.success("✅ **Database (Chuẩn)**")
                    st.write(f"- Họ tên: **{student_db.get('full_name')}**")
                    st.write(f"- MSSV: **{student_db.get('student_id')}**")

            # Cập nhật thông tin chuẩn để ghi Log
            final_info = main_student.copy()
            final_info["Họ và tên"] = student_db.get("full_name")
            final_info["Ngành"] = student_db.get("major")

            # B. Chạy logic Check Vào/Ra (Chống lấy nhầm xe)
            # Hàm này sẽ ghi vào gate_logs hoặc alerts
            res_code, res_msg = check_gate_process(main_plate, mssv)

            if "SUCCESS" in res_code:
                st.success(f"✅ {res_msg}")
                # Khi chạy trên Cloud, ta không lưu cv2.imwrite (vì Cloud sẽ xóa file)
                # Thay vào đó, bạn có thể lưu link ảnh nếu dùng Cloud Storage (tùy chọn)
            else:
                st.error(f"🚨 {res_msg}")
        else:
            st.error(f"❌ Thẻ SV {mssv} chưa có trong hệ thống!")

    elif results_data["students"] and not results_data["plates"]:
        st.warning("📡 Đã nhận diện được Thẻ. Vui lòng di chuyển xe để thấy rõ Biển số!")

    elif results_data["plates"] and not results_data["students"]:
        st.warning("📡 Đã nhận diện được Biển số. Vui lòng đưa Thẻ sinh viên vào vùng quét!")

    return display_img, results_data
# ==========================================
# 4. GIAO DIỆN STREAMLIT
# ==========================================

st.title("VAA Hệ thống giữ xe thẻ sinh viên")
source = st.sidebar.radio("Nguồn đầu vào", ["📷 Camera", "📁 Tải ảnh lên"])

if source == "📁 Tải ảnh lên":
    file = st.file_uploader("Chọn ảnh (Có thể up lần lượt Thẻ rồi đến Biển số)", type=['jpg', 'png', 'jpeg'])

    if st.sidebar.button("🗑️ Xóa lượt quét cũ"):
        st.session_state.pair_data = {"mssv": None, "plate": None, "raw_info": None, "db_info": None}
        st.rerun()

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        res_img, data = process_frame(img)

        # 1. XỬ LÝ KHI CÓ THẺ SV
        if data["students"]:
            current_mssv = data["students"][0]["Mã SV"]
            st.session_state.pair_data["mssv"] = current_mssv
            st.session_state.pair_data["raw_info"] = data["students"][0]
            st.write('Thông tin sinh viên')
            st.table(data['students'])
            # --- ĐỐI CHIẾU DATABASE NGAY LẬP TỨC ---
            st.write("🔍 **Đang đối chiếu danh tính từ Database...**")
            student_db = get_student_from_db(current_mssv)

            if student_db:
                st.session_state.pair_data["db_info"] = student_db
                st.success(f"✅ Tìm thấy SV: {student_db.get('full_name')} - {current_mssv}")

                # HIỂN THỊ BẢNG ĐỐI CHIẾU NGANG (Cái Đức cần ở đây)
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"📝 **OCR đọc được:**\n- Tên: {data['students'][0]['Họ và tên']}\n- MSSV: {current_mssv}")
                with c2:
                    st.success(
                        f"✅ **Database chuẩn:**\n- Tên: {student_db.get('full_name')}\n- MSSV: {student_db.get('student_id')}")
            else:
                st.error(f"❌ Thẻ SV {current_mssv} KHÔNG tồn tại trong Database!")

        # 2. XỬ LÝ KHI CÓ BIỂN SỐ
        if data["plates"]:
            st.session_state.pair_data["plate"] = data["plates"][0]
            st.info(f"📡 Đã nhận diện biển số: {data['plates'][0]}")

        # 3. HIỂN THỊ TRẠNG THÁI TỔNG HỢP
        st.divider()
        st.write("### 🛰️ Trạng thái hệ thống")
        pair = st.session_state.pair_data
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Mã SV", pair["mssv"] if pair["mssv"] else "Chưa có")
        col_m2.metric("Biển số", pair["plate"] if pair["plate"] else "Chưa có")

        # 4. LOGIC GHI VÀO DATABASE (CHỈ CHẠY KHI ĐỦ CẶP)
        if pair["mssv"] and pair["plate"] and pair["db_info"]:
            st.warning("🔄 Đang thực hiện ghi nhận lượt xe Ra/Vào...")
            res_code, res_msg = check_gate_process(pair["plate"], pair["mssv"])

            if "SUCCESS" in res_code:
                st.success(f"🎉 {res_msg}")
                st.balloons()
            else:
                st.error(f"🚨 CẢNH BÁO: {res_msg}")

        st.image(res_img, channels="BGR", caption="Ảnh vừa xử lý")
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

