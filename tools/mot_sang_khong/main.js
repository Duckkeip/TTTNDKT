const fs = require('fs');
const path = require('path');

// Đường dẫn đến thư mục labels (cùng cấp với file main.js)
const labelsDir = path.join(__dirname, 'labels');

function convertLabels() {
    // Kiểm tra xem thư mục labels có tồn tại không
    if (!fs.existsSync(labelsDir)) {
        console.error("LỖI: Không tìm thấy thư mục 'labels' cùng cấp với script!");
        return;
    }

    // Đọc danh sách các file trong thư mục
    fs.readdir(labelsDir, (err, files) => {
        if (err) {
            console.error("Không thể đọc thư mục:", err);
            return;
        }

        const txtFiles = files.filter(file => file.endsWith('.txt'));
        
        if (txtFiles.length === 0) {
            console.log("Không tìm thấy file .txt nào.");
            return;
        }

        txtFiles.forEach(file => {
            const filePath = path.join(labelsDir, file);

            // Đọc nội dung file
            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    console.error(`Lỗi khi đọc file ${file}:`, err);
                    return;
                }

                // Xử lý từng dòng: Nếu bắt đầu bằng '0', đổi thành '1'
                const lines = data.split('\n');
                const newData = lines.map(line => {
                    if (line.startsWith('1')) {
                        return '2' + line.substring(1);
                    }
                    return line;
                }).join('\n');

                // Ghi lại nội dung mới vào file
                fs.writeFile(filePath, newData, 'utf8', (err) => {
                    if (err) {
                        console.error(`Lỗi khi ghi file ${file}:`, err);
                    } else {
                        console.log(`✅ Đã chuyển đổi: ${file}`);
                    }
                });
            });
        });
    });
}

// Chạy hàm
convertLabels();