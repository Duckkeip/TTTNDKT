import splitfolders
import os

# ====== ĐƯỜNG DẪN ======
INPUT_DIR = "dataset"     # chứa images/ và labels/
OUTPUT_DIR = "output"     # nơi xuất train/val/test

# ====== KIỂM TRA ======
if not os.path.exists(INPUT_DIR):
    raise Exception("❌ Không tìm thấy thư mục dataset")

# ====== CHIA DATASET ======
splitfolders.ratio(
    INPUT_DIR,
    output=OUTPUT_DIR,
    seed=42,               # để lần nào chia cũng giống nhau
    ratio=(0.7, 0.2, 0.1), # train / val / test
    group_prefix=None,     # giữ ảnh & label đi cùng nhau
    move=False             # False = copy, True = move
)

print("✅ Chia dataset hoàn tất!")
