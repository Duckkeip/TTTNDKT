const fs = require("fs");
const path = require("path");

const IMG_DIR = "images";
const LABEL_DIR = "labels";
const OUT_IMG = "output/images";
const OUT_LABEL = "output/labels";
const PREFIX = "Otobiendai";

// Tạo thư mục đầu ra
fs.mkdirSync(OUT_IMG, { recursive: true });
fs.mkdirSync(OUT_LABEL, { recursive: true });

// Hàm để trích xuất số từ tên file (ví dụ: "CarLongPlate12" -> 12)
const getNumber = (filename) => {
    const match = filename.match(/\d+/);
    return match ? parseInt(match[0], 10) : -1;
};

// Lấy danh sách file và SẮP XẾP THEO SỐ
const imgFiles = fs.readdirSync(IMG_DIR)
    .filter(img => img.match(/\.(jpg|png|jpeg)$/i))
    .sort((a, b) => getNumber(a) - getNumber(b));

let idx = 1; // Bắt đầu từ 0001 theo yêu cầu của bạn

for (const img of imgFiles) {
    const baseName = path.parse(img).name;
    const ext = path.extname(img);
    const labelFile = baseName + ".txt";
    const labelPath = path.join(LABEL_DIR, labelFile);

    if (!fs.existsSync(labelPath)) {
        console.warn("⚠️ Không tìm thấy label cho:", img);
        continue;
    }

    const newName = `${PREFIX}_${String(idx).padStart(4, "0")}`;

    // Copy ảnh
    fs.copyFileSync(
        path.join(IMG_DIR, img),
        path.join(OUT_IMG, newName + ext)
    );

    // Copy nhãn
    fs.copyFileSync(
        labelPath,
        path.join(OUT_LABEL, newName + ".txt")
    );

    console.log(`🚀 ${img} -> ${newName}${ext}`);
    idx++;
}

console.log(`✅ Hoàn thành! Đã xử lý ${idx - 1} cặp file.`);