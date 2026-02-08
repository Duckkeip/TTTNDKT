import cv2
import numpy as np
import streamlit as st
import easyocr
import re


# Khởi tạo EasyOCR (dùng cache để không load lại nhiều lần)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['vi', 'en'], gpu=False)


reader = load_ocr()


def advanced_enhance(image):
    if image is None or image.size == 0:
        return None

    # 1. Upscaling (Tăng độ phân giải)
    h, w = image.shape[:2]
    upscale_factor = 2
    resized = cv2.resize(image, (w * upscale_factor, h * upscale_factor),
                         interpolation=cv2.INTER_LANCZOS4)

    # 2. Grayscale & CLAHE (Cân bằng tương phản)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 3. Sharpening (Làm sắc nét)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced_gray, -1, kernel)

    # 4. Adaptive Thresholding (Nhị phân hóa)
    thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return {
        "original": image,
        "enhanced_gray": enhanced_gray,
        "final_thresh": thresh
    }


# --- Giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("🔬 So sánh kết quả xử lý ảnh & OCR")

uploaded_file = st.file_uploader("Tải ảnh vùng chọn (ROI) để test OCR", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Đọc ảnh
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Xử lý ảnh
    results = advanced_enhance(img)

    # --- PHẦN LOG OCR ---
    st.subheader("📝 Kết quả OCR thực tế")

    # Tạo 3 cột để test 3 loại ảnh khác nhau
    col_raw, col_gray, col_thresh = st.columns(3)

    with col_raw:
        st.image(results["original"], caption="1. Ảnh Gốc")
        # Đọc thử ảnh gốc
        text_raw = reader.readtext(img, detail=0)
        st.code(f"Dữ liệu đọc được:\n{text_raw}", language="text")

    with col_gray:
        st.image(results["enhanced_gray"], caption="2. Ảnh Enhanced (Grayscale + CLAHE)")
        # Đọc thử ảnh xám đã xử lý
        text_gray = reader.readtext(results["enhanced_gray"], detail=0)
        st.success(f"Dữ liệu đọc được:\n{text_gray}")

    with col_thresh:
        st.image(results["final_thresh"], caption="3. Ảnh Thresh (Nhị phân)")
        # Đọc thử ảnh đen trắng
        text_thresh = reader.readtext(results["final_thresh"], detail=0)
        st.warning(f"Dữ liệu đọc được:\n{text_thresh}")

    # Phân tích kỹ thuật
    with st.expander("🧐 Phân tích kỹ thuật (Nên dùng cái nào?)"):
        st.write("""
        - **Ảnh 1 (Gốc):** Thường bị nhiễu do ánh sáng môi trường, chữ dễ bị dính vào nền.
        - **Ảnh 2 (Enhanced):** Tốt nhất cho **Biển số xe**. Giữ được độ đậm nhạt của chữ nhưng làm rõ nét hơn.
        - **Ảnh 3 (Thresh):** Tốt nhất cho **Thẻ sinh viên** có nền hoa văn phức tạp. Nó xóa sạch màu nền, chỉ để lại hình dáng chữ đen trên nền trắng.
        """)