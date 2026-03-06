# 🚗 Hệ thống AI Giữ xe Thông minh - VAA

Ứng dụng chạy trên nền tảng **Streamlit**, sử dụng mô hình **YOLO** để nhận diện Biển số xe và Thẻ sinh viên, kết hợp với **EasyOCR** để trích xuất thông tin và lưu trữ dữ liệu trên **MongoDB Atlas**.

## ✨ Tính năng chính
- 📷 **Nhận diện thời gian thực:** Quét biển số xe và thẻ sinh viên qua Camera (WebRTC).
- 📁 **Xử lý ảnh tải lên:** Hỗ trợ tải ảnh trực tiếp để kiểm tra.
- 🔍 **Đối chiếu Database:** Tự động kiểm tra thông tin sinh viên từ cơ sở dữ liệu MongoDB.
- 📝 **Ghi nhật ký:** Lưu lịch sử xe ra/vào kèm hình ảnh minh chứng.
- 🚨 **Cảnh báo:** Thông báo nếu thẻ sinh viên không tồn tại hoặc có sai sót.

## 🛠 Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.x
- **Framework:** Streamlit
- **AI/ML:** YOLO (Ultralytics), EasyOCR
- **Database:** MongoDB Atlas
- **Thư viện chính:** OpenCV, Pandas, PyMongo

## 🚀 Hướng dẫn cài đặt (Local)

1. **Clone repository:**
   ```bash
   git clone [https://github.com/ten-user-cua-ban/tttndkt.git](https://github.com/ten-user-cua-ban/tttndkt.git)
   cd tttndkt



TTTNDKT/
├── .streamlit/
│   └── secrets.toml        # (Chỉ dùng ở máy cá nhân, không push lên GitHub)
├── models/
│   ├── Bienso.pt           # Model YOLO nhận diện biển số
│   └── Thesv.pt            # Model YOLO nhận diện thẻ sinh viên
├── app.py                  # File code Python chính (đã sửa WebRTC và MongoDB)
├── requirements.txt        # Danh sách thư viện Python (pip install)
├── packages.txt            # Danh sách thư viện hệ thống (apt-get)
├── README.md               # Hướng dẫn sử dụng dự án
└── .env                    # Lưu MONGO_URI (Dùng ở local, không push lên GitHub)
   
