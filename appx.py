import threading

import pandas as pd
import streamlit as st

import av
import cv2
import numpy as np
import re

import unicodedata
from ultralytics import YOLO
import easyocr
import base64
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from services.auth_service import auth_ui
from twilio.rest import Client

import os

load_dotenv()
#PAYOS
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
expire_time = int(time.time() + 600)
vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
# Khởi tạo bộ nhớ tạm để "ghép cặp" nếu upload nhiều ảnh khác nhau
if 'pair_data' not in st.session_state:
    st.session_state.pair_data = {
        "mssv": None, "plate": None,
        "raw_info": None, "db_info": None,
        "img_sv": None, "img_plate": None  # Thêm 2 dòng này
    }
# Bộ nhớ tạm để lưu kết quả quét được từ Camera
if 'cam_storage' not in st.session_state:
    st.session_state.cam_storage = {"mssv": None, "plate": None, "user_name": None}

if "cam_key" not in st.session_state:
    st.session_state.cam_key = 0
SCAN_COOLDOWN = 4
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = {}

if "upload_pair" not in st.session_state:
    st.session_state.upload_pair = {
        "mssv": None,
        "plate": None,
        "raw_info": None,
        "db_info": None,
        "img_sv": None,
        "img_plate": None
    }



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
        return db["students"], db["gate_logs"], db["alerts"] ,db["recharge_logs"],db["users"]
    except Exception as e:
        st.error(f"❌ Lỗi kết nối MongoDB: {e}")
        st.stop()
students_col, logs_col, alerts_col, recharge_col, users_col = init_db()
db = students_col.database # Bây giờ biến 'db' mới chính thức tồn tại
print(">>> ĐÃ KẾT NỐI DB THÀNH CÔNG, ĐANG QUÉT TOKEN...")
# --- BƯỚC 2: XỬ LÝ KÍCH HOẠT VỚI LOG CHI TIẾT ---
query_params = st.query_params

if "verify_token" in query_params:
    token = query_params["verify_token"]

    # LOG 1: Kiểm tra xem có bắt được Token từ URL không
    st.info(f"🔍 Đang kiểm tra Token từ Email: `{token}`")

    # 1. Chỉ định bảng xử lý
    target_col = users_col  # Dùng luôn biến users_col đã nhận ở trên

    # 2. Tìm User
    user_to_verify = target_col.find_one({"verification_token": token})

    if user_to_verify:
        # LOG 2: Tìm thấy User khớp với Token
        st.write(f"✅ Đã tìm thấy tài khoản: **{user_to_verify.get('username')}**")
        st.write(f"📊 Trạng thái hiện tại (is_active): `{user_to_verify.get('is_active')}`")

        # 3. Thực hiện cập nhật
        result = target_col.update_one(
            {"_id": user_to_verify["_id"]},
            {
                "$set": {"is_active": True},
                "$unset": {"verification_token": ""}
            }
        )

        # LOG 3: Kiểm tra kết quả ghi vào Database
        if result.modified_count > 0:
            st.query_params.clear()
            st.success("🚀 Database đã cập nhật thành công (is_active: True)!")
            st.balloons()

            # window.location.origin sẽ tự lấy https://vaagate.streamlit.app
            # hoặc http://localhost:8501 tùy theo nơi bạn đang đứng
            st.components.v1.html(f"""
                    <script>
                        setTimeout(function(){{
                            window.location.href = window.location.origin; 
                        }}, 3000);
                    </script>
                """, height=0)
            st.stop()
        else:
            st.warning("⚠️ Lệnh Update đã chạy nhưng không có hàng nào bị thay đổi (Có thể is_active đã True sẵn).")
    else:
        # LOG 4: Không tìm thấy User
        st.error("❌ Không tìm thấy User nào có mã xác thực này trong bảng 'users'.")
        # Log thêm để bạn kiểm tra xem mình có lưu nhầm bảng khác không
        st.write("💡 Mẹo: Hãy kiểm tra MongoDB Compass xem Token này nằm ở bảng 'users' hay 'students'.")


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
                                # 2. Lấy dữ liệu mới nhất (bao gồm số dư mới và email) để gửi thông báo
                                updated_user = users_collection.find_one({"student_id": user['student_id']})
                                new_balance = updated_user.get("balance", 0)
                                user_email = updated_user.get("email")

                                from utils.email_service import send_custom_email, get_transaction_template
                                # 3. Gửi Email thông báo qua PayOS
                                if user_email:
                                    html_body = get_transaction_template(
                                        user_name=updated_user.get("full_name", "Sinh viên"),
                                        amount=int(order["amount"]),
                                        balance=new_balance
                                    )
                                    send_custom_email(
                                        receiver_email=user_email,
                                        subject="[VAA Parking] Xác nhận nạp tiền thành công (PayOS)",
                                        html_content=html_body
                                    )

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
        if "checkout_url" not in st.session_state:
            st.session_state.checkout_url = None

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
                        cancelUrl="https://vaagate.streamlit.app/?payment=cancel",
                        expiredAt=expire_time
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
                        "time": datetime.now(vn_tz)
                    })

                    st.session_state.checkout_url = pay_link.checkout_url
                    st.success("✅ Đã tạo mã thanh toán!")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
        if st.session_state.checkout_url:
            st.markdown(f"**Vui lòng quét mã QR bên dưới để hoàn tất:**")
            st.components.v1.iframe(st.session_state.checkout_url, height=700, scrolling=True)

        # Thêm nút để ẩn mã QR nếu khách hàng muốn hủy
            if st.button("Hủy/Đóng mã QR này"):
                st.session_state.checkout_url = None
                st.rerun()

# --- NỘI DUNG CHO ADMIN: THỐNG KÊ ---
if menu == "📊 Thống kê hệ thống":
    st.title("📊 Báo cáo & Thống kê Chuyên sâu")
    @st.fragment(run_every="30s")
    def show_dashboard():
        # --- PHẦN 1: BỘ LỌC (FILTERS) ---
        with st.expander("🔍 Bộ lọc tìm kiếm", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                search_mssv = st.text_input("Mã số sinh viên", placeholder="Nhập MSSV...")
            with col_f2:
                search_plate = st.text_input("Biển số xe", placeholder="Nhập biển số...")
            with col_f3:
                filter_status = st.selectbox("Trạng thái", ["Tất cả", "IN", "OUT"])

        # Xây dựng query cho MongoDB dựa trên bộ lọc
        query = {}
        if search_mssv:
            # Sửa từ "mssv" thành "student_id"
            query["student_id"] = {"$regex": search_mssv, "$options": "i"}
        if search_plate:
            # Sửa từ "plate" thành "plate_detected"
            query["plate_detected"] = {"$regex": search_plate, "$options": "i"}
        if filter_status != "Tất cả":
            query["status"] = filter_status

        # Lấy dữ liệu đã lọc
        filtered_logs = list(logs_col.find(query).sort("time", -1))
        df = pd.DataFrame(filtered_logs)

        # --- PHẦN 2: METRICS (THỐNG KÊ NHANH) ---
        col1, col2, col3, col4 = st.columns(4)

        total_logs = len(df) if not df.empty else 0

        pipeline = [
            {"$sort": {"time": -1}},
            {
                "$group": {
                    "_id": "$plate_detected",  # Tên trường phải khớp với DB của bạn
                    "last_status": {"$first": "$status"}
                }
            },
            {"$match": {"last_status": "IN"}}
        ]

        # Thêm try-except để tránh lỗi vặt làm treo giao diện
        try:
            vehicles_inside = len(list(logs_col.aggregate(pipeline)))
        except:
            vehicles_inside = 0

        # Tính doanh thu: Lưu ý dùng đúng tên trường 'fee_charged' hoặc 'fee' tùy DB của bạn
        # Nếu trong DB lưu là 'fee_charged' thì sửa df['fee'] thành df['fee_charged']
        fee_col = 'fee' if 'fee' in df.columns else ('fee_charged' if 'fee_charged' in df.columns else None)
        total_revenue = df[fee_col].sum() if fee_col and not df.empty else 0

        col1.metric("Tổng lượt (Lọc)", f"{total_logs}")
        col2.metric("Xe trong bãi", f"{vehicles_inside}")
        col3.metric("Doanh thu (Lọc)", f"{total_revenue:,.0f}đ")
        col4.metric("Cảnh báo", f"{alerts_col.count_documents({})}")

        # --- PHẦN 3: BIỂU ĐỒ (VISUALIZATION) ---
        if not df.empty:
            st.subheader("📈 Biểu đồ lưu lượng")
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC").dt.tz_convert("Asia/Ho_Chi_Minh")

            # Biểu đồ đường theo giờ/ngày
            df['hour'] = df['time'].dt.hour
            hourly_counts = df.groupby(['hour', 'status']).size().unstack(fill_value=0)
            st.area_chart(hourly_counts)

        # --- PHẦN 4: BẢNG DỮ LIỆU ---
        st.subheader("📝 Nhật ký chi tiết")

        if not df.empty:
            # 1. Tạo bản sao để xử lý hiển thị
            df_display = df.copy()

            # 2. Xử lý thời gian (Chuyển từ UTC sang Giờ Việt Nam)
            if "time" in df_display.columns:
                df_display["time"] = pd.to_datetime(df_display["time"]).dt.tz_convert("Asia/Ho_Chi_Minh")
                df_display["time"] = df_display["time"].dt.strftime("%d-%m-%Y %H:%M:%S")
            if "fee_charged" in df_display.columns:
                # Chuyển về số (đề phòng dữ liệu dạng chuỗi) và định dạng 1,000 VNĐ
                df_display["fee_charged"] = df_display["fee_charged"].apply(
                    lambda x: f"{x:,.0f} VNĐ" if pd.notnull(x) else "0 VNĐ")
            # 3. Chọn đúng các cột đang có trong DB của bạn
            # Dựa theo ảnh: student_id, student_name, plate_detected, status, fee_charged
            cols_to_show = ["time", "student_id", "student_name", "plate_detected", "status", "fee_charged"]

            # Chỉ lấy những cột thực sự tồn tại để tránh lỗi crash
            existing_cols = [c for c in cols_to_show if c in df_display.columns]
            df_final = df_display[existing_cols]

            # 4. Đổi tên tiêu đề cột cho chuyên nghiệp (Viết tiếng Việt có dấu)
            rename_map = {
                "time": "THỜI GIAN",
                "student_id": "MSSV",
                "student_name": "HỌ TÊN",
                "plate_detected": "BIỂN SỐ",
                "status": "TRẠNG THÁI",
                "fee_charged": "PHÍ (VNĐ)"
            }
            df_final = df_final.rename(columns=rename_map)


            # 5. Định dạng màu sắc dựa trên cột TRẠNG THÁI
            def style_status(val):
                if val == "IN":
                    return "background-color: #d4edda; color: #155724; font-weight: bold;"
                elif val == "OUT":
                    return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
                return ""


            # 6. Hiển thị bảng lên Streamlit
            st.dataframe(
                df_final.style.map(style_status, subset=["TRẠNG THÁI"] if "TRẠNG THÁI" in df_final.columns else []),
                use_container_width=True,
                hide_index=True
            )

            # 7. Nút xuất file (Giữ nguyên MSSV và Biển số trong file CSV)
            csv = df_display.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải báo cáo chi tiết (MSSV & Biển số)",
                data=csv,
                file_name=f"nhat_ky_xe_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Không tìm thấy dữ liệu phù hợp với bộ lọc (MSSV hoặc Biển số).")
    show_dashboard()
#ADMIN: Quản lý Users
def notify_low_balance(student_id, current_balance, user_data):
    """
    Hàm kiểm tra số dư và gửi email cảnh báo nếu số dư dưới ngưỡng 10,000 VNĐ.
    Được gọi khi trừ tiền phí gửi xe hoặc Admin điều chỉnh số dư.
    """
    low_balance_threshold = 10000
    days_between_warnings = 3
    now = datetime.now(vn_tz)

    if current_balance < low_balance_threshold:
        # Lấy mốc thời gian gửi lần cuối từ database
        last_sent = user_data.get("last_warning_sent")

        # Kiểm tra nếu chưa bao giờ gửi hoặc đã quá 3 ngày (tránh spam)
        if not last_sent or (now - last_sent.replace(tzinfo=vn_tz)).days >= days_between_warnings:
            from utils.email_service import send_custom_email, get_low_balance_template

            user_email = user_data.get("email")
            if user_email:
                # Tạo nội dung email từ template
                html_warn = get_low_balance_template(user_data['full_name'], current_balance)

                # Thực hiện gửi email
                sent = send_custom_email(user_email, "[VAA Parking] Cảnh báo số dư tài khoản thấp", html_warn)

                if sent:
                    # Cập nhật lại mốc thời gian gửi vào MongoDB
                    db["users"].update_one(
                        {"student_id": student_id},
                        {"$set": {"last_warning_sent": now}}
                    )
                    print(f">>> Đã gửi email cảnh báo cho {student_id}")
if menu == "👥 Quản lý người dùng":
    st.header("💰 Quản lý Ngân khố & Người dùng")

    # Tạo 2 cột: 1 bên nạp tiền, 1 bên xem danh sách
    col_nap, col_list = st.columns([1, 2])

    with col_nap:
        st.subheader("Nạp tiền vào ví")
        with st.form("recharge_form"):
            target_mssv = st.text_input("Nhập MSSV cần nạp")
            amount = st.number_input("Số tiền nạp (VNĐ)", min_value=-50000, step=1000)
            reason = st.selectbox("Hình thức", ["Chuyển khoản Ngân hàng", "Tiền mặt", "Khuyến mãi"])
            submit_nap = st.form_submit_button("Xác nhận nạp tiền")

            if submit_nap:
                # Kiểm tra user có tồn tại không
                target_user = db["users"].find_one({"student_id": target_mssv})
                if target_user:
                    print(f">>> Đang nạp tiền cho: {target_mssv}")
                    # Lệnh $inc để cộng dồn tiền vào balance
                    db["users"].update_one(
                        {"student_id": target_mssv},
                        {"$inc": {"balance": amount}}
                    )
                    # 2. Lấy số dư mới và gửi mail ngay lập tức
                    updated_user = db["users"].find_one({"student_id": target_mssv})
                    # Gọi hàm thông báo (bạn cần dán hàm notify_low_balance tôi đưa ở câu trước vào appx.py)
                    notify_low_balance(target_mssv, updated_user['balance'], updated_user)

                    new_balance = target_user.get("balance", 0) + amount
                    user_email = target_user.get("email")
                    print(f">>> Email tìm thấy: {user_email}")
                    if user_email:
                        from utils.email_service import send_custom_email, get_transaction_template

                        html_body = get_transaction_template(target_user['full_name'], amount, new_balance)

                        print(">>> Bắt đầu gọi hàm gửi mail...")  # Log 3
                        success = send_custom_email(user_email, "[VAA] Nạp tiền", html_body)

                        if success:
                            print(">>> GỬI MAIL THÀNH CÔNG!")
                        else:
                            print(">>> GỬI MAIL THẤT BẠI - Kiểm tra cấu hình Gmail!")

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
    current_time = datetime.now(vn_tz) # Lưu dạng datetime object để dễ truy vấn sau này

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
    base_path = os.getcwd() # Lấy thư mục hiện tại của dự án
    plate_path = os.path.join(base_path, "models", "Bienso.pt")
    sv_path = os.path.join(base_path, "models", "Thesv.pt")
    # Kiểm tra file có tồn tại không trước khi load

    if not os.path.exists(plate_path) or not os.path.exists(sv_path):
        st.error(f"Không tìm thấy file model tại: {os.path.dirname(plate_path)}")

        st.stop()



    yolo_plate = YOLO(plate_path)
    yolo_sv = YOLO(sv_path)
    reader = easyocr.Reader(
        ['vi', 'en'],
        gpu=False,
        model_storage_directory=os.path.join(base_path, "models")
    )

    return yolo_plate, yolo_sv, reader

@st.cache_resource(ttl=86400)
def get_ice_servers():
    try:
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        print("Twilio timeout, fallback STUN:", e)
        return [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]

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
    resized = cv2.resize(image, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # 2. Tạo bản Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE nhẹ (Giảm clipLimit xuống 1.2 để không bị cháy ảnh)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return {
        "enhanced": binary,  # Thử dùng ảnh nhị phân này cho OCR
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
    now = datetime.now(vn_tz)
    users_col = db["users"]

    user_data = users_col.find_one({"student_id": mssv_ocr})
    if not user_data:
        return "ERROR", "Tài khoản không tồn tại!"

    current_balance = user_data.get("balance", 0)
    is_staff = user_data.get("user_type") == "staff"
    fee = 0 if is_staff else 3000

    # lấy log gần nhất của MSSV này
    last_log = logs_col.find_one(
        {"student_id": mssv_ocr},
        sort=[("time", -1)]
    )

    def clean(p):
        return "".join(filter(str.isalnum, str(p))).upper()

    # ====================
    # XE RA
    # ====================
    if last_log and last_log["status"] == "IN":
        plate_at_in = last_log["plate_detected"]

        # BẮT BUỘC TRÙNG BIỂN SỐ LÚC VÀO
        if clean(plate_detected) != clean(plate_at_in):
            return "ERROR_PLATE", (
                f"Biển số không khớp!\n"
                f"Lúc vào: {plate_at_in}\n"
                f"Lúc ra: {plate_detected}"
            )

        # kiểm tra tiền
        if not is_staff and current_balance < fee:
            return "LOW_BALANCE", "Số dư không đủ!"

        if fee > 0:
            users_col.update_one(
                {"student_id": mssv_ocr},
                {"$inc": {"balance": -fee}}
            )

        logs_col.insert_one({
            "time": now,
            "student_id": mssv_ocr,
            "student_name": user_data["full_name"],
            "plate_detected": plate_detected,
            "status": "OUT",
            "fee_charged": fee
        })

        return "SUCCESS_OUT", "Xe ra bãi thành công"

    # ====================
    # XE VÀO
    # ====================
    else:
        logs_col.insert_one({
            "time": now,
            "student_id": mssv_ocr,
            "student_name": user_data["full_name"],
            "plate_detected": plate_detected,
            "status": "IN"
        })

        return "SUCCESS_IN", "Xe vào bãi thành công"
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
    now = datetime.now(vn_tz)
    os.makedirs("images", exist_ok=True)

    img_name = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
    img_path = f"images/{img_name}"

    with open(img_path, "wb") as f:
        f.write(image_bytes)

    mssv_ocr = raw_info.get("Mã SV", "Không rõ")
    student_db = get_student_from_db(mssv_ocr)

    if not student_db:
        alerts_col.insert_one({
            "time": now,
            "reason": "Student ID not registered",
            "student_ocr": raw_info,
            "plate_detected": plate,
            "image_path": img_path
        })
        return None, False

    def clean_p(p):
        return "".join(filter(str.isalnum, str(p))).upper()

    plate_db = student_db.get("plate", "")
    is_match = clean_p(plate) == clean_p(plate_db)

    if is_match:
        logs_col.insert_one({
            "time": now,
            "student_id": student_db["student_id"],
            "student_name": student_db["full_name"],
            "plate_detected": plate,
            "image_path": img_path,
            "status": "IN",
            "note": "Match plate"
        })
    else:
        alerts_col.insert_one({
            "time": now,
            "student_id": student_db["student_id"],
            "student_name": student_db["full_name"],
            "plate_registered": plate_db,
            "plate_detected": plate,
            "reason": "Plate mismatch",
            "image_path": img_path
        })

    return student_db, is_match

# ==========================================
# 3. HÀM XỬ LÝ CHÍNH (DEEP SCAN)
# ==========================================
def process_frame(img):
    h_orig, w_orig = img.shape[:2]
    display_img = img.copy()

    res_data = {
        "mssv": None, "user_name": None, "plate": None,
        "students": [], "plates": [],
        "crop_sv": None, "crop_plate": None
    }

    # --- 1) QUÉT THẺ SINH VIÊN ---
    sv_results = yolo_sv.predict(img, conf=0.5, imgsz=640, verbose=False, max_det=1)[0]
    for box in sv_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad = 10
        x1, y1, x2, y2 = max(0, x1-pad), max(0, y1-pad), min(w_orig, x2+pad), min(h_orig, y2+pad)
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            res_data["crop_sv"] = crop
            res = advanced_enhance(crop)
            ocr_list = reader.readtext(res["enhanced"], detail=0, paragraph=False)
            info = extract_student_info(ocr_list)
            mssv_found = info.get("Mã SV")
            if mssv_found and mssv_found != "Không rõ":
                clean_mssv = re.sub(r"\D", "", str(mssv_found))
                user_db = db["students"].find_one({"student_id": clean_mssv})
                if user_db:
                    res_data["mssv"] = clean_mssv
                    res_data["user_name"] = user_db.get("full_name")
                    res_data["students"] = [info]
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break

    # --- 2) QUÉT BIỂN SỐ (PHẢI NẰM NGOÀI VÒNG LẶP SV) ---
    plate_results = yolo_plate.predict(img, conf=0.4, imgsz=640, verbose=False)[0]
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad = 5
        x1, y1, x2, y2 = max(0, x1-pad), max(0, y1-pad), min(w_orig, x2+pad), min(h_orig, y2+pad)
        crop_p = img[y1:y2, x1:x2]
        if crop_p.size > 0:
            res_data["crop_plate"] = crop_p
            res_p = advanced_enhance(crop_p)
            ocr_p = reader.readtext(res_p["enhanced"], detail=0)
            raw_plate = "".join(ocr_p).upper()
            fixed_p = re.sub(r'[^A-Z0-9-]', '', raw_plate)
            if len(fixed_p) >= 7:
                res_data["plate"] = fixed_p
                res_data["plates"] = [fixed_p]
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                break

    return display_img, res_data


# --- LỚP XỬ LÝ VIDEO DUY NHẤT ---
class VideoProcessor:
    def __init__(self):
        self.last_data = None
        self.last_frame = None
        self.processing = False
        self.last_detect_time = 0
        self.detect_interval = 1.0
        self.cam_storage = {
            "mssv": None,
            "plate": None,
            "user_name": None
        }

    def detect_async(self, img):
        try:
            small = cv2.resize(img, (640, 360))
            res_img, data = process_frame(small)

            self.last_frame = res_img
            self.last_data = data

            # auto lưu kết quả
            if data.get("mssv") and not self.cam_storage["mssv"]:
                self.cam_storage["mssv"] = data["mssv"]
                self.cam_storage["user_name"] = data["user_name"]

            if data.get("plate") and not self.cam_storage["plate"]:
                self.cam_storage["plate"] = data["plate"]

            # tự động xác nhận luôn khi đủ 2 dữ liệu
            if self.cam_storage["mssv"] and self.cam_storage["plate"]:
                res_code, res_msg = check_gate_process(
                    self.cam_storage["plate"],
                    self.cam_storage["mssv"]
                )

                print("AUTO:", res_msg)

                # giữ UI 3 giây cho người dùng thấy
                time.sleep(3)

                self.cam_storage = {
                    "mssv": None,
                    "plate": None,
                    "user_name": None
                }

        finally:
            self.processing = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if (now - self.last_detect_time > self.detect_interval) and not self.processing:
            self.processing = True
            self.last_detect_time = now

            threading.Thread(
                target=self.detect_async,
                args=(img.copy(),),
                daemon=True
            ).start()

        output = self.last_frame if self.last_frame is not None else img
        return av.VideoFrame.from_ndarray(output, format="bgr24")


# Khởi tạo một bản duy nhất để dùng chung
if "processor" not in st.session_state:
    st.session_state.processor = VideoProcessor()




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
    file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'])

    if st.sidebar.button("🗑️ Xóa lượt quét cũ"):
        st.session_state.upload_pair = {
            "mssv": None, "plate": None, "db_info": None,
            "img_sv": None, "img_plate": None
        }
        st.rerun()

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        res_img, data = process_frame(img)
        pair = st.session_state.upload_pair

        # Gán ảnh thẻ
        if data["crop_sv"] is not None:
            pair["img_sv"] = data["crop_sv"]
        if data["mssv"]:
            pair["mssv"] = data["mssv"]
            student_db = get_student_from_db(data["mssv"])
            if student_db:
                pair["db_info"] = student_db

        # Gán ảnh biển số
        if data["crop_plate"] is not None:
            pair["img_plate"] = data["crop_plate"]
            pair["plate"] = data["plate"]

    # --- HIỂN THỊ ---
    st.divider()
    pair = st.session_state.upload_pair
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🪪 Ảnh Thẻ SV")
        if pair["img_sv"] is not None:
            st.image(cv2.cvtColor(pair["img_sv"], cv2.COLOR_BGR2RGB), use_container_width=True)
            if pair["mssv"]: st.success(f"MSSV: {pair['mssv']}")
        else:
            st.info("Chưa có thẻ")

    with c2:
        st.subheader("📡 Ảnh Biển số")
        if pair["img_plate"] is not None:
            st.image(cv2.cvtColor(pair["img_plate"], cv2.COLOR_BGR2RGB), use_container_width=True)
            if pair["plate"]: st.success(f"Biển số: {pair['plate']}")
        else:
            st.info("Chưa có biển số")

    # --- GHI DATABASE (GATELOG) ---
    if pair["mssv"] and pair["plate"] and pair.get("db_info"):
        st.write("---")
        if st.button("🚀 XÁC NHẬN GHI LOG & TRỪ TIỀN (3.000đ)", type="primary", use_container_width=True):
            # Gọi hàm xử lý gate đã có sẵn trong code của bạn (trừ tiền + logs_col)
            res_code, res_msg = check_gate_process(pair["plate"], pair["mssv"])

            if "SUCCESS" in res_code:
                st.success(f"🎉 {res_msg}")
                st.balloons()
                # Reset để chuẩn bị quét lượt mới
                st.session_state.upload_pair = {"mssv": None, "plate": None, "db_info": None, "img_sv": None,
                                                "img_plate": None}
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"🚨 Lỗi: {res_msg}")

# --- TRƯỜNG HỢP 2: DÙNG CAMERA (WEBRTC) ---
else:

    col_cam, col_info = st.columns([7, 3])

    with col_cam:
        st.info("📷 Màn hình Giám sát")
        if st.button("🔄 Khởi động lại Camera"):
            st.session_state.cam_key += 1
            st.session_state.processor = VideoProcessor()
            st.rerun()
        ctx = webrtc_streamer(
            key=f"parking-ai-{st.session_state.cam_key}",
            video_frame_callback=st.session_state.processor.recv,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 360},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True,
        )

        # Thêm dòng này để "nhắc" Streamlit giữ thread ổn định
        if not ctx.state.playing:
            st.session_state.processor = VideoProcessor()

    with col_info:
        @st.fragment(run_every="0.5s")
        def status_ui():
            st.subheader("📑 Trạng thái realtime")

            processor = st.session_state.processor
            storage = processor.cam_storage

            # Hiển thị MSSV
            if storage["mssv"]:
                st.success(f"🪪 MSSV: {storage['mssv']}")
                st.info(f"👤 {storage['user_name']}")
            else:
                st.warning("👉 Đang chờ quét thẻ sinh viên...")

            # Hiển thị biển số
            if storage["plate"]:
                st.success(f"📡 Biển số: {storage['plate']}")
            else:
                if storage["mssv"]:
                    st.warning("👉 Đã có MSSV, đang chờ biển số...")
                else:
                    st.warning("👉 Chưa quét được MSSV...")


        status_ui()