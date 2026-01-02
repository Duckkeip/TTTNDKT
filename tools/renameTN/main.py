import os
import shutil

img_dir = "images"
label_dir = "labels"

out_img_dir = "output/images"
out_label_dir = "output/labels"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

prefix = "Otobiendai"   # 🔴 đổi tên TẠI ĐÂY

i = 0
for filename in sorted(os.listdir(img_dir)):
    if filename.lower().endswith(".jpg"):
        old_img = os.path.join(img_dir, filename)
        old_label = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        new_name = f"{prefix}_{i:04d}"

        shutil.copy(old_img, os.path.join(out_img_dir, new_name + ".jpg"))

        if os.path.exists(old_label):
            shutil.copy(old_label, os.path.join(out_label_dir, new_name + ".txt"))
        else:
            print(f"⚠️ Thiếu label: {filename}")

        i += 1

print("✅ Xuất file mới với tên do bạn đặt xong")
