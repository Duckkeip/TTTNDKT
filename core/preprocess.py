import cv2
import numpy as np

def enhance_for_ocr(crop):
    if crop is None or crop.size == 0:
        return crop

    # Bước 1: Phóng to ảnh lên gấp đôi bằng phép nội suy Cubic (mượt hơn)
    # Ảnh to giúp EasyOCR nhận diện các ký tự nhỏ tốt hơn
    h, w = crop.shape[:2]
    enhanced = cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # Bước 2: Khử nhiễu nhẹ (Denoising)
    # Giúp loại bỏ các hạt li ti mà EasyOCR hay nhầm thành dấu chấm hoặc chữ E
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    return enhanced