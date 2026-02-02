# api_server.py
from fastapi import FastAPI
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
    now = datetime.now()

    plate = data.get("plate")
    student_info = data.get("student")
    image_base64 = data.get("image")

    # Lưu ảnh
    os.makedirs("images", exist_ok=True)
    img_name = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
    img_path = f"images/{img_name}"

    with open(img_path, "wb") as f:
        f.write(base64.b64decode(image_base64))

    # ==== LOGIC CHÍNH THEO THẺ SV ====
    if not student_info or student_info.get("MSSV") == "Không rõ":
        alerts_col.insert_one({
            "time": now,
            "reason": "Student card not recognized",
            "student_ocr": student_info,
            "plate_detected": plate,
            "image_path": img_path
        })
        return {"status": "ALERT_CARD"}

    student_db = students_col.find_one({"student_id": student_info["MSSV"]})

    if not student_db:
        alerts_col.insert_one({
            "time": now,
            "reason": "Student ID not registered",
            "student_ocr": student_info,
            "plate_detected": plate,
            "image_path": img_path
        })
        return {"status": "ALERT_UNKNOWN_STUDENT"}

    note = "Match plate" if plate == student_db["plate"] else "Plate mismatch"

    logs_col.insert_one({
        "time": now,
        "student": {
            "student_id": student_db["student_id"],
            "name": student_db["name"]
        },
        "plate_detected": plate,
        "plate_registered": student_db["plate"],
        "image_path": img_path,
        "status": "IN",
        "note": note
    })

    return {"status": "OK"}
