import streamlit as st
import hashlib
from datetime import datetime


def hash_password(password):
    """Mã hóa mật khẩu để không lưu văn bản thô"""
    return hashlib.sha256(str.encode(password)).hexdigest()


def auth_ui(db):
    """Giao diện Đăng nhập / Đăng ký"""
    users_col = db["users"]

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
                        "created_at": datetime.now()
                    }
                    users_col.insert_one(user_payload)
                    st.success("Đăng ký thành công! Hãy chuyển sang tab Đăng nhập.")