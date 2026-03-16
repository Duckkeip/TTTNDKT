import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import random
import string

def send_custom_email(receiver_email, subject, html_content):
    """
    Hàm gửi email dùng chung, lấy cấu hình từ Streamlit Secrets.
    """
    try:
        # Lấy thông tin bảo mật từ Secrets
        sender_email = st.secrets["GMAIL_USER"]
        app_password = st.secrets["GMAIL_PASSWORD"]

        # Tạo đối tượng message
        msg = MIMEMultipart()
        msg['From'] = f"VAA Smart Parking <{sender_email}>"
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Gắn nội dung HTML
        msg.attach(MIMEText(html_content, 'html'))

        # Kết nối server và gửi
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Lỗi gửi email: {e}")
        return False
#OTP reset MK
def generate_random_password(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def get_transaction_template(user_name, amount, balance):
    """
    Tạo template HTML cho thông báo nạp tiền.
    """
    return f"""
    <div style="font-family: sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 10px;">
        <h2 style="color: #007bff;">Xác nhận nạp tiền thành công</h2>
        <p>Xin chào <b>{user_name}</b>,</p>
        <p>Hệ thống giữ xe VAA thông báo giao dịch của bạn đã hoàn tất:</p>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">Số tiền nạp:</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{amount:,.0f} VNĐ</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">Số dư mới:</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold; color: #28a745;">{balance:,.0f} VNĐ</td>
            </tr>
        </table>
        <p style="margin-top: 20px; font-size: 0.9em; color: #666;">Đây là email tự động, vui lòng không phản hồi.</p>
    </div>
    """

def get_low_balance_template(user_name, balance):
    """
    Tạo template HTML cho cảnh báo số dư thấp.
    """
    return f"""
    <div style="font-family: sans-serif; border: 1px solid #ffc107; padding: 20px; border-radius: 10px; background-color: #fffdf5;">
        <h2 style="color: #856404;">⚠️ Cảnh báo số dư tài khoản thấp</h2>
        <p>Xin chào <b>{user_name}</b>,</p>
        <p>Hệ thống VAA Smart Parking nhận thấy số dư trong tài khoản của bạn hiện tại không còn nhiều:</p>
        <div style="background-color: #eee; padding: 15px; border-radius: 5px; text-align: center; margin: 20px 0;">
            <span style="font-size: 1.2em;">Số dư hiện tại: </span>
            <b style="font-size: 1.5em; color: #dc3545;">{balance:,.0f} VNĐ</b>
        </div>
        <p>Để đảm bảo quá trình ra vào bãi xe không bị gián đoạn, vui lòng nạp thêm tiền vào tài khoản.</p>
        <p style="margin-top: 20px; font-size: 0.9em; color: #666;">Đây là email nhắc nhở định kỳ khi số dư dưới ngưỡng quy định.</p>
    </div>
    """