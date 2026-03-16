import streamlit as st
import hashlib
from datetime import datetime
import time


@st.dialog("🔐 Khôi phục mật khẩu")
def password_reset_dialog(users_col):
    # Khởi tạo các biến session state
    if "reset_step" not in st.session_state:
        st.session_state.reset_step = 1
    if "otp_expiry" not in st.session_state:
        st.session_state.otp_expiry = 0

    # BƯỚC 1: NHẬP EMAIL
    if st.session_state.reset_step == 1:
        st.write("Vui lòng nhập Email để nhận mã xác thực (OTP).")
        email = st.text_input("Email VAA đăng ký", placeholder="example@vaa.edu.vn")

        if st.button("Gửi mã OTP", use_container_width=True, type="primary"):
            user = users_col.find_one({"email": email})
            if user:
                import random
                otp = str(random.randint(100000, 999999))
                st.session_state.otp_code = otp
                st.session_state.reset_email = email
                # Thiết lập hết hạn sau 120 giây (2 phút)
                st.session_state.otp_expiry = time.time() + 120

                # --- GIAO DIỆN EMAIL ĐẸP ---
                subject = f"{otp} là mã xác thực bãi xe của bạn"
                html_content = f"""
                <div style="font-family: Arial, sans-serif; max-width: 400px; margin: auto; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px;">
                    <div style="text-align: center;">
                        <h2 style="color: #007bff; margin-bottom: 10px;">Xác thực tài khoản</h2>
                        <p style="color: #666;">Mã OTP khôi phục mật khẩu của bạn là:</p>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; font-size: 24px; font-weight: bold; color: #333; letter-spacing: 5px; margin: 20px 0;">
                            {otp}
                        </div>
                        <p style="font-size: 12px; color: #999;">Mã này sẽ hết hạn sau <b>2 phút</b>.</p>
                        <p style="font-size: 12px; color: #999;">Nếu không phải bạn yêu cầu, vui lòng bỏ qua email này.</p>
                    </div>
                </div>
                """

                with st.spinner("Đang gửi mã..."):
                    from utils.email_service import send_custom_email
                    if send_custom_email(email, subject, html_content):
                        st.session_state.reset_step = 2
                        st.rerun()
                    else:
                        st.error("Lỗi gửi mail. Vui lòng kiểm tra lại cấu hình!")
            else:
                st.error("Email này không tồn tại trong hệ thống!")

    # BƯỚC 2: XÁC THỰC OTP
    elif st.session_state.reset_step == 2:
        # Tính thời gian còn lại
        time_left = int(st.session_state.otp_expiry - time.time())

        if time_left > 0:
            st.write(f"Mã đã được gửi đến: **{st.session_state.reset_email}**")
            st.info(f"Mã sẽ hết hạn sau: **{time_left} giây**")

            otp_in = st.text_input("Nhập mã OTP (6 số)", maxlength=6)

            if st.button("Xác nhận", use_container_width=True, type="primary"):
                if otp_in == st.session_state.otp_code:
                    st.session_state.reset_step = 3
                    st.rerun()
                else:
                    st.error("Mã OTP không chính xác!")

            if st.button("Gửi lại mã", variant="ghost"):
                st.session_state.reset_step = 1
                st.rerun()
        else:
            st.error("Mã OTP đã hết hạn!")
            if st.button("Gửi lại mã mới", use_container_width=True):
                st.session_state.reset_step = 1
                st.rerun()

    # BƯỚC 3: ĐẶT MẬT KHẨU MỚI
    elif st.session_state.reset_step == 3:
        st.success("Xác thực thành công! Nhập mật khẩu mới bên dưới.")
        new_p = st.text_input("Mật khẩu mới", type="password")
        conf_p = st.text_input("Xác nhận mật khẩu", type="password")

        if st.button("Cập nhật mật khẩu", use_container_width=True, type="primary"):
            if new_p == conf_p and len(new_p) >= 6:
                users_col.update_one(
                    {"email": st.session_state.reset_email},
                    {"$set": {"password": hash_password(new_p)}}
                )
                st.balloons()
                st.success("Đã đổi mật khẩu thành công!")
                time.sleep(2)
                st.session_state.reset_step = 1  # Reset trạng thái cho lần sau
                st.rerun()
            else:
                st.error("Mật khẩu không khớp hoặc ngắn hơn 6 ký tự!")

def hash_password(password):
    """Mã hóa mật khẩu để không lưu văn bản thô"""
    return hashlib.sha256(str.encode(password)).hexdigest()
def auth_ui(db):
    """Giao diện Đăng nhập / Đăng ký"""
    users_col = db["users"]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "login"
    tab1, tab2 = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập (MSSV)")
            password = st.text_input("Mật khẩu", type="password")
            submit = st.form_submit_button("Đăng nhập")

            if submit:
                user = users_col.find_one({"username": username, "password": hash_password(password)})
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user
                    st.success(f"Chào mừng {user['full_name']} quay trở lại!")
                    st.rerun()
                else:
                    st.error("Sai tên đăng nhập hoặc mật khẩu!")

            # Đặt nút Quên mật khẩu ở đây (Ngay dưới Form)
        col1, col2 = st.columns([4, 1])
        with col2:  # Đẩy nút sang bên phải cho tinh tế
            if st.button("Quên mật khẩu?", key="forgot_btn"):
                password_reset_dialog(users_col)

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("MSSV")
            new_email = st.text_input("Email VAA")
            new_name = st.text_input("Họ và tên")
            new_pass = st.text_input("Mật khẩu mới", type="password")
            confirm_pass = st.text_input("Xác nhận mật khẩu", type="password")
            reg_submit = st.form_submit_button("Đăng ký tài khoản")

            if reg_submit:
                if new_pass != confirm_pass:
                    st.warning("Mật khẩu xác nhận không khớp!")
                elif users_col.find_one({"username": new_user}):
                    st.error("Tài khoản này đã tồn tại!")
                else:
                    user_payload = {
                        "username": new_user,
                        "student_id": new_user,
                        "password": hash_password(new_pass),
                        "email": new_email,
                        "full_name": new_name,
                        "balance": 0,  # Ngân khố khởi tạo là 0đ
                        "role": "user",
                        "created_at": datetime.now(),
                        "user_type":"student"
                    }
                    users_col.insert_one(user_payload)
                    st.success("Đăng ký thành công! Đang chuyển sang đăng nhập...")

                    st.session_state.active_tab = "login"
                    st.rerun()
