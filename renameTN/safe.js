const fs = require("fs");
const path = require("path");

// ===== ĐƯỜNG DẪN =====
const BASE_DIR = __dirname;

const IMG_DIR   = path.join(BASE_DIR, "images");
const LABEL_DIR = path.join(BASE_DIR, "labels");

const OUT_IMG   = path.join(BASE_DIR, "output", "images");
const OUT_LABEL = path.join(BASE_DIR, "output", "labels");

const PREFIX = "Cobao";

// ===== TẠO THƯ MỤC ĐẦU RA =====
fs.mkdirSync(OUT_IMG, { recursive: true });
fs.mkdirSync(OUT_LABEL, { recursive: true });

// ===== HÀM LẤY SỐ TRONG TÊN FILE =====
const getNumber = (filename) => {
    const match = filename.match(/\d+/);
    return match ? parseInt(match[0], 10) : 0;
};

// ===== KIỂM TRA THƯ MỤC =====
if (!fs.existsSync(IMG_DIR)) {
    console.warn("⚠️ Không tìm thấy thư mục images → không có ảnh để xử lý");
}

if (!fs.existsSync(LABEL_DIR)) {
    console.warn("⚠️ Không tìm thấy thư mục labels → sẽ copy ảnh không kèm label");
}

// ===== ĐỌC FILE ẢNH (KHÔNG CRASH) =====
const imgFiles = fs.existsSync(IMG_DIR)
    ? fs.readdirSync(IMG_DIR)
        .filter(f => f.match(/\.(jpg|jpeg|png)$/i))
        .sort((a, b) => getNumber(a) - getNumber(b))
    : [];

let idx = 1;
let copiedImg = 0;
let copiedLabel = 0;

// ===== RENAME + COPY =====
for (const img of imgFiles) {
    const baseName = path.parse(img).name;
    const ext = path.extname(img);

    const labelPath = path.join(LABEL_DIR, baseName + ".txt");

    const newName = `${PREFIX}_${String(idx).padStart(4, "0")}`;

    // ===== COPY ẢNH (LUÔN COPY) =====
    fs.copyFileSync(
        path.join(IMG_DIR, img),
        path.join(OUT_IMG, newName + ext)
    );
    copiedImg++;

    // ===== COPY LABEL NẾU CÓ =====
    if (fs.existsSync(labelPath)) {
        fs.copyFileSync(
            labelPath,
            path.join(OUT_LABEL, newName + ".txt")
        );
        copiedLabel++;
    } else {
        console.warn(`⚠️ Thiếu label: ${img}`);
    }

    console.log(`✅ ${img} → ${newName}${ext}`);
    idx++;
}

// ===== TỔNG KẾT =====
console.log("\n===== HOÀN THÀNH =====");
console.log(`🖼️ Ảnh đã copy: ${copiedImg}`);
console.log(`🏷️ Label đã copy: ${copiedLabel}`);
console.log("📁 Output:", path.join(BASE_DIR, "output"));
