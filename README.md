
<img width="1578" height="376" alt="image" src="https://github.com/user-attachments/assets/b3b4425f-92d0-423d-86e2-3e9ffb9fd0e8" />

#  Hệ thống AI Giữ xe Thông minh - VAA

Hệ thống quản lý bãi xe thông minh dành cho Học viện Hàng không Việt Nam (VAA). Ứng dụng tích hợp AI để tự động hóa quy trình nhận diện biển số và đối soát thẻ sinh viên, giúp tăng cường an ninh và giảm thời gian chờ đợi.



##  Tính năng chính

- 📷 **Nhận diện thời gian thực:** Tích hợp Camera qua WebRTC để quét biển số xe và thẻ sinh viên trực tiếp.
- 📁 **Xử lý ảnh tải lên:** Hỗ trợ tải tệp hình ảnh để kiểm tra thủ công.
- 🔍 **Đối chiếu Database:** Tự động truy vấn và xác thực thông tin sinh viên từ **MongoDB Atlas**.
- 📝 **Ghi nhật ký:** Lưu lịch sử xe ra/vào với đầy đủ mốc thời gian và hình ảnh minh chứng.
- 🚨 **Hệ thống cảnh báo:** Thông báo ngay lập tức nếu thẻ sinh viên không hợp lệ hoặc không có trong hệ thống.

## 🛠 Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.x
- **Giao diện:** [Streamlit](https://streamlit.io/)
- **Trí tuệ nhân tạo:** - **YOLO (Ultralytics):** Nhận diện vùng chứa biển số và thẻ sinh viên.
- **EasyOCR:** Trích xuất ký tự từ vùng ảnh đã nhận diện.
- **Cơ sở dữ liệu:** MongoDB Atlas (Cloud Database).
- **Chức năng thanh toán:** PayOS.
- **Thư viện bổ trợ:** OpenCV, Pandas, PyMongo, Streamlit-WebRTC.

## 📂 Cấu trúc dự án

```text
TTTNDKT/
├── .streamlit/
│   └── secrets.toml      # Cấu hình bảo mật (Không push public)
├── models/
│   ├── Bienso.pt         # Model YOLO nhận diện biển số
│   └── Thesv.pt          # Model YOLO nhận diện thẻ sinh viên
├── app.py                # Mã nguồn chính của ứng dụng
├── requirements.txt      # Danh sách thư viện Python (pip)
├── packages.txt          # Thư viện hệ thống (dùng cho Streamlit Cloud)
├── README.md             # Hướng dẫn dự án
└── .env                  # Lưu biến môi trường MONGO_URI (Không push public)
```
## WORKFLOW
```mermaid
    flowchart TD
    Start([Bắt đầu quét]) --> Input[Camera quét Thẻ & Biển số]
    Input --> AI_Proc{AI Xử lý - YOLO + OCR}
    
    AI_Proc -->|Thất bại| Retry[Yêu cầu quét lại]
    AI_Proc -->|Thành công| DB_Check{Kiểm tra Database}
    
    DB_Check -->|MSSV không tồn tại| Error1[Báo lỗi: SV chưa đăng ký]
    DB_Check -->|MSSV Hợp lệ| Mode{Kiểm tra trạng thái xe}
    
    Mode -->|Xe đang ở Ngoài| CheckIn[Lệnh: Cho xe VÀO]
    Mode -->|Xe đang ở Trong| CheckOut[Lệnh: Cho xe RA]
    
    CheckIn --> SaveIN[(Ghi log vào MongoDB: IN)]
    CheckOut --> Balance{Kiểm tra số dư}
    
    %% Luồng Nạp tiền & Gmail 1
    Balance -->|Không| PayOS[Nạp tiền PayOS]
    PayOS --> Mail1[Gmail: Xác nhận nạp tiền]
    Mail1 --> Balance
    
    %% Luồng Trừ tiền & Gmail 2
    Balance -->|Có| Pay[Trừ tiền & Ghi log vào MongoDB: OUT]
    Pay --> Mail2[Gmail: Thông báo biến động số dư]
    
    SaveIN --> End([Hoàn tất])
    Mail2 --> End
    
    %% Định dạng màu sắc cho nổi bật
    style Mail1 fill:#brown,stroke:#333,stroke-width:2px
    style Mail2 fill:#gray,stroke:#333,stroke-width:2px
    style PayOS fill:#blue,stroke:#333,stroke-width:2px
```
```mermaid
sequenceDiagram
    actor Student as Sinh viên (User)
    participant Web as Web (Streamlit)
    participant Server as Server (Backend)
    participant PayOS as Server PayOS
    participant Bank as Ngân hàng
    participant DB as Database

    Student->>Web: Đăng nhập & Chọn phương thức nạp tiền
    Web->>Web: Lưu session
    
    Student->>Web: Nhập số tiền cần nạp
    Web->>Server: Gửi thông tin nạp tiền
    
    Server->>PayOS: Gọi API tạo đơn hàng (payment-requests)
    PayOS->>Bank: Yêu cầu tạo mã QR-Pay
    Bank-->>PayOS: Trả về mã QR-Pay cho đơn hàng
    PayOS-->>Server: Trả về thông tin thanh toán (QR Code link)
    
    Server->>DB: Lưu hoá đơn với status "Đang chờ"
    Server-->>Web: Hiển thị mã QR lên giao diện
    
    Note over Student, Bank: Giai đoạn thực hiện quét mã
    
    alt Sinh viên ấn Hủy
        Student->>Web: Nhấn nút Hủy
        Web-->>Student: Quay lại trang chọn phương thức
    else Sinh viên quét mã thanh toán
        Student->>Bank: Thực hiện chuyển tiền qua App Ngân hàng
        Bank-->>PayOS: Cập nhật thanh toán mới (Webhook)
        PayOS-->>Server: Xác nhận hoàn tất thanh toán (Status: PAID)
        
        Server->>DB: Update hoá đơn thành "Đã trả"
        Server->>DB: Thêm vào lịch sử giao dịch & Cộng số dư sinh viên
        
        Server-->>Web: Thông báo thành công
        Web-->>Student: Chuyển hướng đến trang "Thanh toán thành công"
    end
```
### HOST tại: **[https://vaagate.streamlit.app/](https://vaagate.streamlit.app/)**
