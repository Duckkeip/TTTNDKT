import easyocr
import re

class OCRProcessor:
    def __init__(self):
        # Khởi tạo một lần duy nhất để tiết kiệm RAM
        self.reader = easyocr.Reader(['vi', 'en'], gpu=False)

    def read_plate(self, roi):
        results = self.reader.readtext(roi, detail=0)
        # Nối các phần tử lại
        plate_text = "".join(results).upper()

        # Xóa mọi ký tự không phải chữ và số
        plate_text = re.sub(r'[^0-9A-Z]', '', plate_text)

        # Danh sách đen: Những thứ chắc chắn không phải biển số xe Việt Nam
        blacklist = ["EEE", "EEEE", "IIII", "XXXX", "NAO", "ABC"]
        for trash in blacklist:
            plate_text = plate_text.replace(trash, "")

        # Biển số VN thường có ít nhất 4 số, nếu ngắn quá thường là nhiễu
        if len(plate_text) < 4:
            return ""

        return plate_text

    def read_student_card(self, roi):
        """Đọc danh sách văn bản trên thẻ sinh viên"""
        # Sửa lỗi: bỏ bớt '.reader'
        results = self.reader.readtext(roi, detail=0)
        return results