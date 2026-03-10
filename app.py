import pandas as pd
import streamlit as st
import cv2
import numpy as np
import re
import os
import unicodedata
from ultralytics import YOLO
import easyocr
import base64
from pymongo import MongoClient
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from services.auth_service import auth_ui

load_dotenv()
from payos import PayOS
from payos.types import CreatePaymentLinkRequest
import pytz
from datetime import datetime, timedelta
# Import theo đường dẫn tuyệt đối này để lách lỗi 'cannot import name'

payos = PayOS(
    client_id=os.getenv("PAYOS_CLIENT_ID"),
    api_key=os.getenv("PAYOS_API_KEY"),
    checksum_key=os.getenv("PAYOS_CHECKSUM_KEY")
)
order_code = int(datetime.now().timestamp() * 1000)
vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
# Khởi tạo bộ nhớ tạm để "ghép cặp" nếu upload nhiều ảnh khác nhau
if 'pair_data' not in st.session_state:
    st.session_state.pair_data = {"mssv": None, "plate": None, "raw_info": None}


@st.cache_resource
def init_db():
    uri = None

    # Kiểm tra xem có đang chạy trên Cloud (Streamlit Secrets) không
    try:
        if "MONGO_URI" in st.secrets:
            uri = st.secrets["MONGO_URI"]
    except:
        # Nếu không có Secrets (đang chạy Local), bỏ qua và tìm ở .env
        pass

    # Nếu trên Cloud không có, thì tìm ở file .env (Local)
    if not uri:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        uri = os.getenv("MONGO_URI")

    if not uri:
        st.error("❌ Không tìm thấy MONGO_URI trong cả Secrets và file .env!")
        st.stop()

    try:
        client = MongoClient(uri)
        db = client["TN"]
        return db["students"], db["gate_logs"], db["alerts"]
    except Exception as e:
        st.error(f"❌ Lỗi kết nối MongoDB: {e}")
        st.stop()
# QUAN TRỌNG: Gán biến ở đây để các hàm khác như get_student_from_db có thể dùng được
students_col, logs_col, alerts_col = init_db()
db = students_col.database

# Khởi tạo các biến session cần thiết
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
# --- 2. KIỂM TRA ĐĂNG NHẬP (Gộp logic để tránh văng app) ---
# Nếu chưa logged_in HOẶC chưa có user_info, hiển thị form đăng nhập
# ===== Khôi phục session sau khi PayOS redirect =====
params = st.query_params

if "payment" in params and "sid" in params:
    sid = params["sid"]

    user_data = db["users"].find_one({"student_id": sid})

    if user_data:
        st.session_state.logged_in = True
        st.session_state.user_info = user_data
if not st.session_state.logged_in or st.session_state.user_info is None:
    auth_ui(db) # Truyền db vào để thực hiện query login
    st.stop() # Dừng tại đây, không chạy các dòng code bên dưới

user = st.session_state.user_info

# Tìm các đơn hàng PENDING của user này
# --- LOGIC KIỂM TRA VÀ CỘNG TIỀN (SỬA LẠI) ---
# Lấy tất cả lịch sử nạp tiền của user này (không chỉ PENDING) để hiển thị đầy đủ
all_orders = list(db["recharge_logs"].find({
    "student_id": user['student_id']
}).sort("time", -1))  # Sắp xếp đơn mới nhất lên đầu
with st.expander("📜 Lịch sử giao dịch", expanded=True):
    # Tạo bộ lọc ngày tháng
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        start_date = st.date_input("Từ ngày", value=datetime.now(vn_tz) - timedelta(days=7))
    with col_f2:
        end_date = st.date_input("Đến ngày", value=datetime.now())
    # Reset trang về 1 nếu thay đổi bộ lọc
    if 'last_start_date' not in st.session_state or st.session_state.last_start_date != start_date:
        st.session_state.current_page = 1
        st.session_state.last_start_date = start_date
    # 2. Lấy dữ liệu từ MongoDB
    # Chuyển đổi start_date và end_date sang datetime để query
    query = {
        "student_id": user['student_id'],
        "time": {
            "$gte": datetime.combine(start_date, datetime.min.time()),
            "$lte": datetime.combine(end_date, datetime.max.time())
        }
    }

    all_orders = list(db["recharge_logs"].find(query).sort("time", -1))
    if all_orders:
            # Tạo danh sách để hiển thị bảng
            display_data = []
            # Bảng ánh xạ trạng thái sang tiếng Việt
            status_map = {
                "PAID": "✅ ĐÃ TRẢ",
                "PENDING": "⏳ ĐANG CHỜ",
                "CANCELLED": "❌ HỦY",
                "EXPIRED": "⚠️ HẾT HẠN"
            }

            for order in all_orders:
                raw_time = order["time"]
                now = datetime.now()
                current_status = order["status"]
                # Không thực hiện astimezone nữa để tránh bị cộng thêm 7 tiếng
                vn_time_display = raw_time.strftime("%d/%m/%Y %H:%M:%S")
                # Nếu đơn đang PENDING, phải hỏi PayOS xem khách đã HỦY chưa
                if current_status == "PENDING" and now - raw_time < timedelta(minutes=15):
                    try:
                        # Dùng đúng hàm get_payment_link_info hoặc getPaymentLinkInformation
                        payos_info = payos.payment_requests.get(order["orderCode"])
                        st.write("PAYOS STATUS:", payos_info.status)
                        if payos_info.status in ["CANCELLED", "EXPIRED"]:
                            # Cập nhật ngay vào DB để lần sau không cần hỏi lại PayOS nữa
                            current_status = payos_info.status

                            db["recharge_logs"].update_one(
                                {"orderCode": order["orderCode"]},
                                {"$set": {"status": current_status}}
                            )

                        elif payos_info.status == "PAID":
                            # Thực hiện cộng tiền vào MongoDB bảng 'users'
                            users_collection = students_col.database["users"]
                            result = users_collection.update_one(
                                {"student_id": user['student_id']},
                                {"$inc": {"balance": int(order["amount"])}}
                            )

                            if result.modified_count > 0:
                                db["recharge_logs"].update_one(
                                    {"orderCode": order["orderCode"]},
                                    {"$set": {"status": "PAID"}}
                                )
                                current_status = "PAID"
                                st.session_state.user_info["balance"] += int(order["amount"])
                                st.toast(f"💰 Đã cộng {order['amount']:,} VNĐ!", icon="✅")
                                st.rerun()
                    except:
                        pass  # Nếu lỗi API PayOS thì bỏ qua để hiện PENDING tiếp
                display_data.append({
                    "Mã đơn hàng": str(order["orderCode"]),
                    "Ngày giao dịch": vn_time_display,
                    "Số tiền": f"{order['amount']:,} VNĐ",
                    "Tình trạng": status_map.get(current_status, current_status)
                })

            df = pd.DataFrame(display_data)

            rows_per_page = 5
            total_rows = len(df)
            total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)

            # Hiển thị bảng
            start_idx = (st.session_state.current_page - 1) * rows_per_page
            st.dataframe(
                df.iloc[start_idx: start_idx + rows_per_page],
                use_container_width=True,
                hide_index=True,
                column_config={"Mã đơn hàng": st.column_config.TextColumn("Mã đơn hàng")}  # Căn trái
            )

            # --- LOGIC PHÂN TRANG RÚT GỌN (1 ... 10 11 12 ... 100) ---
            if total_pages > 1:
                st.write("---")


                # Hàm xác định các số trang cần hiển thị
                def get_page_range(current, total):
                    if total <= 7:
                        return list(range(1, total + 1))

                    pages = [1]
                    if current > 3:
                        pages.append("...")

                    # Hiển thị các trang xung quanh trang hiện tại
                    for i in range(max(2, current - 1), min(total, current + 2)):
                        pages.append(i)

                    if current < total - 2:
                        pages.append("...")

                    pages.append(total)
                    return pages


                page_range = get_page_range(st.session_state.current_page, total_pages)

                # Tạo các cột để căn giữa (Cột trống - Cụm nút - Cột trống)
                # Tỉ lệ [2, 6, 2] giúp cụm nút ở giữa rộng và cân đối
                empty_l, center_col, empty_r = st.columns([1, 8, 1])

                with center_col:
                    # Tạo số lượng cột tương ứng với các nút cần hiện (Prev + Page Range + Next)
                    n_cols = len(page_range) + 2
                    btn_cols = st.columns(n_cols)

                    # 1. Nút Trước (Prev)
                    with btn_cols[0]:
                        if st.button("⬅️", disabled=(st.session_state.current_page == 1), use_container_width=True):
                            st.session_state.current_page -= 1
                            st.rerun()

                    # 2. Các nút số trang và dấu "..."
                    for idx, pg in enumerate(page_range):
                        with btn_cols[idx + 1]:
                            if pg == "...":
                                st.write("<p style='text-align:center; padding-top:5px;'>...</p>",
                                         unsafe_allow_html=True)
                            else:
                                # Đổi màu nút nếu là trang hiện tại
                                btn_type = "primary" if pg == st.session_state.current_page else "secondary"
                                if st.button(str(pg), type=btn_type, use_container_width=True):
                                    st.session_state.current_page = pg
                                    st.rerun()

                    # 3. Nút Sau (Next)
                    with btn_cols[-1]:
                        if st.button("➡️", disabled=(st.session_state.current_page == total_pages),
                                     use_container_width=True):
                            st.session_state.current_page += 1
                            st.rerun()

                st.markdown(
                    f"<p style='text-align: center; color: gray;'>Đang xem trang {st.session_state.current_page} / {total_pages}</p>",
                    unsafe_allow_html=True)
    else:
        st.info("Chưa có lịch sử nạp tiền trong khoảng thời gian này.")

# 3. CẤU HÌNH GIAO DIỆN (Đoạn này chỉ chạy khi đã vượt qua bước 2)
st.set_page_config(page_title="Hệ thống AI Giữ xe VAA", layout="wide")

st.sidebar.markdown(f"### 👤 {user['full_name']}")
# Chỉ sinh viên mới hiện số dư ví

def get_user_balance():
    user = db["users"].find_one({"student_id": st.session_state.user_info["student_id"]})
    st.session_state.user_info = user
    return user["balance"]
balance = get_user_balance()
st.sidebar.markdown(f"💳 **Số dư:** `{balance:,}` VNĐ")
st.autorefresh(interval=7000)

# Hiển thị loại tài khoản
u_type = "Cán bộ/Giảng viên" if user.get("user_type") == "staff" else "Sinh viên"
st.sidebar.info(f"🏷️ Loại: {u_type}")


# --- PHÂN QUYỀN GIAO DIỆN ---
if user.get("role") == "admin":
    menu = st.sidebar.radio("Chức năng Admin", [ "📊 Thống kê hệ thống", "👥 Quản lý người dùng"])
else:
    menu = "📜 Lịch sử cá nhân"
    # Nút đăng xuất
    if st.sidebar.button("🚪 Đăng xuất"):
        # Xóa trạng thái đăng nhập
        st.session_state.logged_in = False
        st.session_state.user_info = None
        # Xóa các dữ liệu tạm thời khác nếu có
        if 'pair_data' in st.session_state:
            del st.session_state.pair_data

        st.success("Đã đăng xuất thành công!")
        st.rerun()  # Tải lại trang để quay về màn hình đăng nhập

# Nếu là User bình thường, dừng các logic quét ở dưới và chỉ hiện lịch sử
if menu == "📜 Lịch sử cá nhân":
    st.title("📜 Lịch sử cá nhân")

    # Tab hiển thị: 1 bên là Lịch sử ra vào, 1 bên là Nạp tiền
    tab1, tab2 = st.tabs(["🚗 Lịch sử ra vào", "💳 Nạp tiền vào ví"])

    with tab1:
        my_logs = list(logs_col.find({"student_id": user["student_id"]}).sort("time", -1))
        if my_logs:
            st.dataframe(pd.DataFrame(my_logs).drop(columns=["_id"]), use_container_width=True)
        else:
            st.info("Chưa có lịch sử ra vào.")

    with tab2:

        st.subheader("Nạp tiền tự động qua QR (Ngân hàng)")

        with st.form("payment_form"):
            amount = st.number_input("Số tiền (Min 3,000đ)", min_value=3000, step=1000)

            if st.form_submit_button("Tạo mã thanh toán"):
                try:
                    order_code = int(datetime.now().timestamp() * 1000)
                    final_amount = int(amount)

                    payment_data = CreatePaymentLinkRequest(
                        orderCode=order_code,
                        amount=final_amount,
                        description=f"NAPTIEN {user['student_id']}"[:25],
                        returnUrl="https://vaagate.streamlit.app/?payment=success",
                        cancelUrl="https://vaagate.streamlit.app/?payment=cancel"
                    )

                    # API mới của PayOS
                    pay_link = payos.payment_requests.create(payment_data)

                    checkout_url = pay_link.checkout_url

                    # Lưu log
                    db["recharge_logs"].insert_one({
                        "orderCode": order_code,
                        "student_id": user['student_id'],
                        "amount": final_amount,
                        "status": "PENDING",
                        "time": datetime.now()
                    })

                    st.success("✅ Đã tạo mã thanh toán!")

                    st.markdown(f"**Vui lòng quét mã QR bên dưới để hoàn tất:**")
                    st.components.v1.iframe(checkout_url, height=700, scrolling=True)
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
# --- NỘI DUNG CHO ADMIN: THỐNG KÊ ---
if menu == "📊 Thống kê hệ thống":
    st.title("📊 Báo cáo & Thống kê")

    col1, col2, col3 = st.columns(3)
    # Lấy dữ liệu thực tế từ DB
    total_logs = logs_col.count_documents({})
    total_in = logs_col.count_documents({"status": "IN"})
    total_alerts = alerts_col.count_documents({})

    col1.metric("Tổng lượt xe", f"{total_logs} lượt")
    col2.metric("Xe đang trong bãi", f"{total_in} xe")
    col3.metric("Cảnh báo vi phạm", f"{total_alerts} vụ", delta_color="inverse")

    st.subheader("📝 Nhật ký ra vào mới nhất")
    all_logs = list(logs_col.find().sort("time", -1).limit(50))
    if all_logs:
        st.dataframe(pd.DataFrame(all_logs).drop(columns=["_id"]))
#ADMIN: Quản lý Users
if menu == "👥 Quản lý người dùng":
    st.header("💰 Quản lý Ngân khố & Người dùng")

    # Tạo 2 cột: 1 bên nạp tiền, 1 bên xem danh sách
    col_nap, col_list = st.columns([1, 2])

    with col_nap:
        st.subheader("Nạp tiền vào ví")
        with st.form("recharge_form"):
            target_mssv = st.text_input("Nhập MSSV cần nạp")
            amount = st.number_input("Số tiền nạp (VNĐ)", min_value=1000, step=1000)
            reason = st.selectbox("Hình thức", ["Chuyển khoản Ngân hàng", "Tiền mặt", "Khuyến mãi"])
            submit_nap = st.form_submit_button("Xác nhận nạp tiền")

            if submit_nap:
                # Kiểm tra user có tồn tại không
                target_user = db["users"].find_one({"student_id": target_mssv})
                if target_user:
                    # Lệnh $inc để cộng dồn tiền vào balance
                    db["users"].update_one(
                        {"student_id": target_mssv},
                        {"$inc": {"balance": amount}}
                    )
                    # Ghi lại lịch sử nạp tiền vào collection giao dịch (tùy chọn)
                    st.success(f"✅ Đã nạp {amount:,} VNĐ cho SV {target_user['full_name']}")
                    st.balloons()
                else:
                    st.error("❌ không tìm thấy sinh viên này!")

    with col_list:
        st.subheader("Danh sách người dùng")
        # 1. Lấy dữ liệu từ MongoDB
        all_users = list(db["users"].find({}, {"password": 0}))

        if all_users:
            df_users = pd.DataFrame(all_users)

            # 2. ĐỊNH NGHĨA BẢN ĐỒ ĐỔI TÊN (Key cũ: Key mới)
            # Hãy đảm bảo các Key bên trái khớp chính xác với field trong MongoDB của bạn
            name_map = {
                "student_id": "MSSV",
                "full_name": "Họ và tên",
                "user_type": "Loại",
                "balance": "Số tiền"
            }

            # 3. KIỂM TRA VÀ BỔ SUNG CỘT THIẾU (Tránh lỗi nếu DB trống)
            for old_key in name_map.keys():
                if old_key not in df_users.columns:
                    df_users[old_key] = 0 if old_key == "balance" else "N/A"

            # 4. THỰC HIỆN ĐỔI TÊN
            df_display = df_users.rename(columns=name_map)

            # 5. CHỌN CÁC CỘT ĐÃ ĐỔI TÊN ĐỂ HIỂN THỊ
            cols_to_show = ["MSSV", "Họ và tên", "Loại", "Số tiền"]
            st.dataframe(df_display[cols_to_show], use_container_width=True)
        else:
            st.info("Chưa có người dùng nào.")


def send_to_api(frame, plate, student_info):
    """
    Ghi trực tiếp vào MongoDB Atlas thay vì gọi qua localhost
    """
    current_time = datetime.now() # Lưu dạng datetime object để dễ truy vấn sau này

    # 1. Xử lý ảnh (Giữ nguyên logic của bạn)
    h, w = frame.shape[:2]
    if w > 1000:
        new_w = 800
        new_h = int(h * (new_w / w))
        frame = cv2.resize(frame, (new_w, new_h))

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    success, buffer = cv2.imencode('.jpg', frame, encode_param)

    if not success:
        st.error("❌ Lỗi mã hóa ảnh!")
        return

    img_base64 = base64.b64encode(buffer).decode()

    # 2. Chuẩn bị Payload để lưu thẳng vào DB
    payload = {
        "plate": plate,
        "student": student_info,
        "image": img_base64,
        "time": current_time,
        "status": "LOG_ENTRY"
    }

    try:
        # Ghi thẳng vào collection logs_col đã khởi tạo ở trên
        logs_col.insert_one(payload)
        st.toast(f"✅ Đã lưu dữ liệu: {plate}", icon="📡")
    except Exception as e:
        st.error(f"⚠️ Lỗi lưu Database: {str(e)}")
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
    now = datetime.now()
    users_col = db["users"]  # Sử dụng collection users mới

    # 1. Tìm thông tin người dùng trong collection users
    user_data = users_col.find_one({"student_id": mssv_ocr})
    if not user_data:
        return "ERROR", f"Tài khoản {mssv_ocr} không tồn tại trên hệ thống!"

    # 2. Xác định phí (Cán bộ miễn phí, Sinh viên 3000đ)
    is_staff = (user_data.get("user_type") == "staff")
    fee = 0 if is_staff else 3000

    # 3. Tìm lượt VÀO (IN) gần nhất
    last_entry = logs_col.find_one(
        {"student_id": mssv_ocr, "status": "IN"},
        sort=[("time", -1)]
    )

    def clean(p):
        return "".join(filter(str.isalnum, str(p))).upper()

    # --- TRƯỜNG HỢP: XE ĐANG RA (OUT) ---
    if last_entry:
        plate_at_in = last_entry.get("plate_detected")
        if clean(plate_detected) == clean(plate_at_in):

            # Kiểm tra tiền nếu là sinh viên
            if not is_staff and user_data["balance"] < fee:
                return "LOW_BALANCE", f"Số dư không đủ ({user_data['balance']:,} VNĐ). Cần 3,000 VNĐ để ra!"

            # Thực hiện trừ tiền trong DB
            if fee > 0:
                users_col.update_one({"student_id": mssv_ocr}, {"$inc": {"balance": -fee}})

            # Ghi log RA
            logs_col.insert_one({
                "time": now,
                "student_id": mssv_ocr,
                "student_name": user_data["full_name"],
                "plate_detected": plate_detected,
                "status": "OUT",
                "fee_charged": fee,
                "note": "Ra bãi thành công"
            })
            return "SUCCESS_OUT", f"MỜI RA! Phí: {fee:,} VNĐ. Số dư còn lại: {user_data['balance'] - fee:,} VNĐ"
        else:
            return "ALERT_THEFT", f"⚠️ SAI BIỂN SỐ! Vào: {plate_at_in} - Ra: {plate_detected}"

    # --- TRƯỜNG HỢP: XE ĐANG VÀO (IN) ---
    else:
        logs_col.insert_one({
            "time": now,
            "student_id": mssv_ocr,
            "student_name": user_data["full_name"],
            "plate_detected": plate_detected,
            "status": "IN",
            "note": "Vào bãi"
        })
        return "SUCCESS_IN", f"MỜI VÀO! Chào {user_data['full_name']} ({u_type})"
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


# Tạo một lớp để xử lý luồng Video
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Gọi hàm xử lý AI của bạn
        res_img, data = process_frame(img)

        # Trả về khung hình đã được vẽ khung nhận diện
        return res_img


# ==========================================
# 4. CẤU CẤU HÌNH WEBRTC (Đưa ra ngoài để tối ưu)
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_data = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Gọi hàm xử lý AI
        res_img, data = process_frame(img)
        self.last_data = data
        return cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)


RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# 5. GIAO DIỆN CHÍNH
# ==========================================
user = st.session_state.user_info

# Kiểm tra nếu role KHÔNG PHẢI admin (Dựa trên ảnh MongoDB của bạn)
if user.get("role") != "admin":

    # Hiển thị số dư cho sinh viên xem thay vì form quét thẻ



    # Dừng app tại đây để sinh viên không thấy phần Camera/Upload bên dưới
    st.stop()

# --- NẾU LÀ ADMIN, TIẾP TỤC HIỂN THỊ GIAO DIỆN CHÍNH ---
st.title("VAA Hệ thống giữ xe thẻ sinh viên")


source = st.sidebar.radio("Nguồn đầu vào", ["📷 Camera", "📁 Tải ảnh lên"])

# --- TRƯỜNG HỢP 1: TẢI ẢNH LÊN ---
if source == "📁 Tải ảnh lên":
    # (Giữ nguyên toàn bộ code xử lý File Uploader của bạn ở đây...)
    file = st.file_uploader("Chọn ảnh (Có thể up lần lượt Thẻ rồi đến Biển số)", type=['jpg', 'png', 'jpeg'])

    if st.sidebar.button("🗑️ Xóa lượt quét cũ"):
        st.session_state.pair_data = {"mssv": None, "plate": None, "raw_info": None, "db_info": None}
        st.rerun()

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        res_img, data = process_frame(img)

        # Xử lý logic Thẻ SV
        if data["students"]:
            current_mssv = data["students"][0]["Mã SV"]
            st.session_state.pair_data["mssv"] = current_mssv
            st.session_state.pair_data["raw_info"] = data["students"][0]

            student_db = get_student_from_db(current_mssv)
            if student_db:
                st.session_state.pair_data["db_info"] = student_db
                st.success(f"✅ Tìm thấy SV: {student_db.get('full_name')}")
            else:
                st.error(f"❌ Thẻ SV {current_mssv} KHÔNG tồn tại trong Database!")

        # Xử lý logic Biển số
        if data["plates"]:
            st.session_state.pair_data["plate"] = data["plates"][0]
            st.info(f"📡 Đã nhận diện biển số: {data['plates'][0]}")

        # Hiển thị Trạng thái & Ghi Database
        st.divider()
        pair = st.session_state.pair_data
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Mã SV", pair["mssv"] if pair["mssv"] else "Chưa có")
        col_m2.metric("Biển số", pair["plate"] if pair["plate"] else "Chưa có")

        if pair["mssv"] and pair["plate"] and pair["db_info"]:
            res_code, res_msg = check_gate_process(pair["plate"], pair["mssv"])
            if "SUCCESS" in res_code:
                st.success(f"🎉 {res_msg}")
                st.balloons()
            else:
                st.error(f"🚨 CẢNH BÁO: {res_msg}")

        st.image(res_img, caption="Ảnh vừa xử lý")

# --- TRƯỜNG HỢP 2: DÙNG CAMERA (WEBRTC) ---
else:
    st.info("💡 Hướng dẫn: Đưa Thẻ SV hoặc Biển số vào trước Camera.")

    ctx = webrtc_streamer(
        key="parking-ai",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        if st.checkbox("Hiển thị nhật ký quét thời gian thực"):
            data_now = ctx.video_processor.last_data
            if data_now:
                if data_now["plates"]:
                    st.success(f"📡 Biển số: {data_now['plates'][0]}")
                if data_now["students"]:
                    st.table(data_now["students"])
