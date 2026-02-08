from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from datetime import datetime
import os
import base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["TN"]

students_col = db["students"]
logs_col = db["gate_logs"]
alerts_col = db["alerts"]


@app.post("/api/gate-event")
async def receive_event(data: dict):
    try:
        now = datetime.now()

        # Lấy dữ liệu từ Payload
        plate = data.get("plate", "unknown")
        student_info = data.get("student")
        image_base64 = data.get("image")

        # 1. Kiểm tra dữ liệu đầu vào cơ bản
        if not image_base64:
            raise HTTPException(status_code=400, detail="Missing image data")

        # 2. Lưu ảnh vật lý
        os.makedirs("images", exist_ok=True)
        img_name = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
        img_path = f"images/{img_name}"

        try:
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
        except Exception as e:
            print(f"Error saving image: {e}")
            img_path = "error_path"

        # 3. LOGIC XỬ LÝ THEO KHUÔN MẪU MỚI
        # Chú ý: Dùng đúng key "Mã SV" thay vì "MSSV"
        mssv_ocr = student_info.get("Mã SV") if student_info else "Không rõ"

        if not student_info or mssv_ocr == "Không rõ":
            alerts_col.insert_one({
                "time": now,
                "reason": "Student card not recognized",
                "student_ocr": student_info,
                "plate_detected": plate,
                "image_path": img_path
            })
            return {"status": "ALERT_CARD", "message": "OCR failed to read Student ID"}

        # 4. Truy vấn Database theo MSSV
        student_db = students_col.find_one({"student_id": mssv_ocr})

        if not student_db:
            alerts_col.insert_one({
                "time": now,
                "reason": "Student ID not registered",
                "student_ocr": student_info,
                "plate_detected": plate,
                "image_path": img_path
            })
            return {"status": "ALERT_UNKNOWN_STUDENT", "message": f"ID {mssv_ocr} not in DB"}

        # 5. So khớp biển số
        # Chuẩn hóa biển số để so sánh (xóa khoảng trắng, gạch ngang)
        def clean_p(p):
            return "".join(filter(str.isalnum, str(p))).upper()

        is_match = clean_p(plate) == clean_p(student_db.get("plate", ""))
        note = "Match plate" if is_match else "Plate mismatch"

        # 6. Ghi log vào Database
        logs_col.insert_one({
            "time": now,
            "student": {
                "student_id": student_db["student_id"],
                "name": student_db.get("name", student_info.get("Họ và tên", "Unknown"))
            },
            "plate_detected": plate,
            "plate_registered": student_db.get("plate"),
            "image_path": img_path,
            "status": "IN",
            "note": note
        })

        return {"status": "OK", "is_match": is_match}

    except Exception as e:
        # Log lỗi chi tiết ra console để debug
        print(f"🚨 SERVER ERROR: {str(e)}")
        return {"status": "ERROR", "message": str(e)}