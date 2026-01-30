import re
import unicodedata

class StudentCardService:
    def __init__(self):
        self.majors = ["CÔNG NGHỆ THÔNG TIN", "QUẢN TRỊ KINH DOANH", "KỸ THUẬT HÀNG KHÔNG", "QUẢN LÝ HOẠT ĐỘNG BAY"]

    def normalize_vietnamese(self, text):
        if not text: return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text.upper().replace('Đ', 'D')

    def extract_info(self, ocr_results):
        # Đảm bảo ocr_results là list string
        if not ocr_results:
            return {"MSSV": "Trống", "Họ Tên": "Trống", "Ngành": "Trống"}

        full_text = " | ".join(ocr_results).upper()
        no_accent_text = self.normalize_vietnamese(full_text)

        # Khớp key với logic hiển thị st.table([student]) trong app.py
        info = {
            "MSSV": "Trống",
            "Họ Tên": "Trống",
            "Ngành": "Trống"
        }

        # 1. Tìm MSSV (Dãy số từ 8-11 ký tự)
        id_match = re.search(r'\b\d{8,11}\b', no_accent_text)
        if id_match:
            info["MSSV"] = id_match.group()

        # 2. Tìm Ngành học
        if any(x in no_accent_text for x in ["CONG NGHE", "TIN HOC", "CNTT"]):
            info["Ngành"] = "Công nghệ thông tin"
        elif "QUAN TRI" in no_accent_text:
            info["Ngành"] = "Quản trị kinh doanh"
        elif "KY THUAT" in no_accent_text:
            info["Ngành"] = "Kỹ thuật hàng không"
        elif "VAN HANH" in no_accent_text or "QUAN LY BAY" in no_accent_text:
            info["Ngành"] = "Quản lý hoạt động bay"

        # 3. Tìm Họ tên (Cải tiến Regex)
        # Thường tên sinh viên trên thẻ VAA viết in hoa và đứng riêng một dòng hoặc sau nhãn
        name_pattern = re.search(r'(?:HO TEN|SINH VIEN|NAME)[:\s\-|]+([A-Z\s]{5,})', no_accent_text)
        if name_pattern:
            name = name_pattern.group(1).strip()
            # Loại bỏ các từ khóa bị dính vào tên
            for trash in ["MSSV", "NGANH", "KHOA", "NIEN KHOA", "|"]:
                name = name.split(trash)[0]
            info["Họ Tên"] = name.strip()

        return info