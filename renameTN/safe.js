const fs = require("fs");
const path = require("path");

// ===== FIX ĐƯỜNG DẪN (QUAN TRỌNG) =====
const BASE_DIR = __dirname;

const IMG_DIR   = path.join(BASE_DIR, "images");
const LABEL_DIR = path.join(BASE_DIR, "labels");

const OUT_IMG   = path.join(BASE_DIR, "output", "images");
const OUT_LABEL = path.join(BASE_DIR, "output", "labels");

const PREFIX = "Xemaybiento";

// ===== TẠO THƯ MỤC ĐẦU RA =====
fs.mkdirSync(OUT_IMG, { recursive: true });
fs.mkdirSync(OUT_LABEL, { recursive: true });

// ===== HÀM LẤY SỐ TRONG TÊN FILE =====
const getNumber = (filename) => {
    const match = filename.match(/\d+/);
    return match ? parseInt(match[0], 10) : -1;
};

// ===== ĐỌC & SẮP XẾP FILE ẢNH =====
if (!fs.existsSync(IMG_DIR)) {
    console.error("❌ Không tìm thấy thư mục images:", IMG_DIR);
    process.exit(1);
}

if (!fs.existsSync(LABEL_DIR)) {
    console.error("❌ Không tìm thấy thư mục labels:", LABEL_DIR);
    process.exit(1);
}

const imgFiles = fs.readdirSync(IMG_DIR)
    .filter(img => img.match(/\.(jpg|jpeg|png)$/i))
    .sort((a, b) => getNumber(a) - getNumber(b));

let idx = 1;

// ===== XỬ LÝ RENAME + COPY =====
for (const img of imgFiles) {
    const baseName = path.parse(img).name;
    const ext = path.extname(img);

    const labelPath = path.join(LABEL_DIR, baseName + ".txt");

    if (!fs.existsSync(labelPath)) {
        console.warn("⚠️ Không tìm thấy label cho:", img);
        continue;
    }

    const newName = `${PREFIX}_${String(idx).padStart(4, "0")}`;

    fs.copyFileSync(
        path.join(IMG_DIR, img),
        path.join(OUT_IMG, newName + ext)
    );

    fs.copyFileSync(
        labelPath,
        path.join(OUT_LABEL, newName + ".txt")
    );

    console.log(`🚀 ${img} → ${newName}${ext}`);
    idx++;
}

console.log(`✅ Hoàn thành! Đã xử lý ${idx - 1} cặp file.`);
