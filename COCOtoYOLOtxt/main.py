import json
import os


def convert_coco_to_yolo(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Map image_id -> info
    images = {img['id']: img for img in data['images']}

    # 🔥 TẠO CATEGORY MAP ĐÚNG
    categories = data['categories']
    category_map = {cat['id']: idx for idx, cat in enumerate(categories)}

    print("Category mapping:")
    for cat in categories:
        print(f"  {cat['name']} -> class {category_map[cat['id']]}")

    for ann in data['annotations']:
        img_info = images.get(ann['image_id'])
        if not img_info:
            continue

        img_w = img_info['width']
        img_h = img_info['height']
        img_filename = os.path.splitext(img_info['file_name'])[0]

        x_min, y_min, w, h = ann['bbox']

        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # ✅ CLASS ID CHUẨN
        class_id = category_map[ann['category_id']]

        label_path = os.path.join(output_dir, f"{img_filename}.txt")
        with open(label_path, 'a') as f_out:
            f_out.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            )

    print("✅ Convert COCO → YOLO hoàn tất!")


convert_coco_to_yolo(
    'labels_thesvcobao_2025-12-31-10-57-21.json',
    'labels'
)
