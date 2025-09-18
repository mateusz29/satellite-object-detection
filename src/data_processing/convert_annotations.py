import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

FAIR1M_CLASSES = {
    "A220": 0,
    "A321": 0,
    "A330": 0,
    "A350": 0,
    "ARJ21": 0,
    "Boeing737": 0,
    "Boeing747": 0,
    "Boeing777": 0,
    "Boeing787": 0,
    "C919": 0,
    "other-airplane": 0,
    "Bridge": 1,
    "Intersection": 2,
    "Roundabout": 3,
    "Bus": 4,
    "Cargo-Truck": 4,
    "Dump-Truck": 4,
    "Small-Car": 4,
    "Trailer": 4,
    "Truck-Tractor": 4,
    "Van": 4,
    "Dry-Cargo-Ship": 5,
    "Fishing-Boat": 5,
    "Engineering-Ship": 5,
    "Liquid-Cargo-Ship": 5,
    "Motorboat": 5,
    "Passenger-Ship": 5,
    "Tugboat": 5,
    "Warship": 5,
    "other-ship": 5,
}

DOTAV2_CLASSES = {
    "plane": 0,
    "bridge": 1,
    "roundabout": 3,
    "large-vehicle": 4,
    "small-vehicle": 4,
    "ship": 5,
}

DIORR_CLASSES = {
    "Airplane": 0,
    "Airport": 1,
    "Baseball field": 2,
    "Basketball court": 3,
    "Bridge": 4,
    "Chimney": 5,
    "Dam": 6,
    "Expressway service area": 7,
    "Expressway toll station": 8,
    "Golf course": 9,
    "Ground track field": 10,
    "Harbor": 11,
    "Overpass": 12,
    "Ship": 13,
    "Stadium": 14,
    "Storage tank": 15,
    "Tennis court": 16,
    "Train station": 17,
    "Vehicle": 18,
    "Wind mill": 19,
}

MSGO_CLASSES = {
    "Plane": 0,
    "Bridge": 1,
    "Intersection": 2,
    "Roundabout": 3,
    "Vehicle": 4,
    "Ship": 5,
}

MSGO_CLASSES_REVERSED = {v: k for k, v in MSGO_CLASSES.items()}


def convert_fair1m_to_yolo(line: str, img_width: int, img_height: int) -> str:
    # YOLO OBB: class_index x1 y1 x2 y2 x3 y3 x4 y4
    # FAIR1M label: 1275 458 1494 88 1417 43 1199 414 Liquid-Cargo-Ship 1
    parts = line.strip().split()

    class_name = parts[-2]
    class_index = FAIR1M_CLASSES.get(class_name, -1)

    if class_index == -1:
        return ""

    coords = list(map(int, parts[:-2]))
    norm_coords = []

    for i, val in enumerate(coords):
        norm_val = val / img_width if i % 2 == 0 else val / img_height
        norm_coords.append(norm_val)

    return f"{class_index} " + " ".join(f"{c:.6f}" for c in norm_coords)


def convert_dotav2_to_yolo(line: str, img_width: int, img_height) -> str:
    # DOTAv2 label: 1076.0 2972.0 1082.0 2976.0 1072.0 2991.0 1065.0 2985.0 small-vehicle 0
    parts = line.strip().split()

    class_name = parts[-2]
    class_index = DOTAV2_CLASSES.get(class_name, -1)

    if class_index == -1:
        return ""

    coords = list(map(float, parts[:-2]))
    norm_coords = []

    for i, val in enumerate(coords):
        norm_val = val / img_width if i % 2 == 0 else val / img_height
        norm_coords.append(norm_val)

    return f"{class_index} " + " ".join(f"{c:.6f}" for c in norm_coords)


def convert_dior_to_yolo(line: str) -> str:
    parts = line.strip().split()

    class_index = int(parts[0])

    mapping = {0: 0, 4: 1, 18: 4, 13: 5}
    if class_index not in mapping:
        return ""

    new_class_index = mapping[class_index]

    coords = list(map(float, parts[1:]))

    return f"{new_class_index} " + " ".join(f"{c:.6f}" for c in coords)


def walkdir_fair1m_and_convert(path: str) -> None:
    fair1m_path = Path(path)
    train_images_path = fair1m_path / "train" / "images"
    train_labels_path = fair1m_path / "train" / "labelTxt"
    val_images_path = fair1m_path / "val" / "images"
    val_labels_path = fair1m_path / "val" / "labelTxt"

    for split, img_dir, label_dir in [
        ("train", train_images_path, train_labels_path),
        ("val", val_images_path, val_labels_path),
    ]:
        for label_file in label_dir.glob("*.txt"):
            img_file = img_dir / (label_file.stem + ".jpg")

            with Image.open(img_file) as img:
                w, h = img.size

            out_label_dir = fair1m_path / split / "labels"
            out_label_dir.mkdir(exist_ok=True)
            out_file = out_label_dir / label_file.name

            with open(label_file) as lf, open(out_file, "w") as of:
                for line in lf:
                    yolo_line = convert_fair1m_to_yolo(line, w, h)
                    if yolo_line:
                        of.write(yolo_line + "\n")


def walkdir_dotav2_and_convert(path: str) -> None:
    dotav2_path = Path(path)
    train_images_path = dotav2_path / "images" / "train"
    train_labels_path = dotav2_path / "labels" / "train_original"
    val_images_path = dotav2_path / "images" / "val"
    val_labels_path = dotav2_path / "labels" / "val_original"

    for split, img_dir, label_dir in [
        ("train", train_images_path, train_labels_path),
        ("val", val_images_path, val_labels_path),
    ]:
        for label_file in label_dir.glob("*.txt"):
            img_file = img_dir / (label_file.stem + ".jpg")

            with Image.open(img_file) as img:
                w, h = img.size

            out_label_dir = dotav2_path / split / "labels"
            out_label_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_label_dir / label_file.name

            with open(label_file) as lf, open(out_file, "w") as of:
                for line in lf:
                    yolo_line = convert_dotav2_to_yolo(line, w, h)
                    if yolo_line:
                        of.write(yolo_line + "\n")


def walkdir_dior_and_convert(path: str) -> None:
    dior_path = Path(path)
    train_labels_path = dior_path / "train" / "labels"
    val_labels_path = dior_path / "val" / "labels"
    test_labels_path = dior_path / "test" / "labels"

    for label_dir in [train_labels_path, val_labels_path, test_labels_path]:
        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                yolo_line = convert_dior_to_yolo(line)
                if yolo_line:
                    new_lines.append(yolo_line + "\n")

            with open(label_file, "w") as f:
                f.writelines(new_lines)


def yolo_obb_to_coco(
    yolo_coords_norm: list[float], img_width: int, img_height: int
) -> tuple[list[float], list[float], float]:
    points_norm = np.array(yolo_coords_norm).reshape(4, 2)
    points_abs = points_norm * np.array([img_width, img_height])
    coco_segmentation = [round(coord, 2) for corner in points_abs for coord in corner]

    x_min, y_min = np.min(points_abs, axis=0)
    x_max, y_max = np.max(points_abs, axis=0)

    coco_bbox = [
        round(float(x_min), 2),
        round(float(y_min), 2),
        round(float(x_max - x_min), 2),
        round(float(y_max - y_min), 2),
    ]

    x = points_abs[:, 0]
    y = points_abs[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return coco_segmentation, coco_bbox, round(float(area), 2)


def create_master_coco_json(root_dir: str) -> None:
    coco_data = {
        "info": {"description": "Pre-sliced dataset"},
        "licenses": [],
        "categories": [
            {"id": cid, "name": cname, "supercategory": "object"} for cid, cname in MSGO_CLASSES_REVERSED.items()
        ],
        "images": [],
        "annotations": [],
    }

    root_path = Path(root_dir)
    image_id_counter, annotation_id_counter = 1, 1
    skipped_annotations_count = 0

    images_dir = root_path / "images"
    labels_dir = root_path / "labels"

    label_files = labels_dir.glob("*.txt")

    for label_file in tqdm(label_files, desc="Processing"):
        img_file = images_dir / f"{label_file.stem}.jpg"

        with Image.open(img_file) as img:
            img_width, img_height = img.size

        image_info = {
            "id": image_id_counter,
            "file_name": img_file.relative_to(root_path).as_posix(),
            "width": img_width,
            "height": img_height,
        }
        coco_data["images"].append(image_info)

        with open(label_file) as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()

            if len(parts) != 9:
                continue

            class_id = int(parts[0])
            yolo_obb_data = [float(p) for p in parts[1:]]

            segmentation, bbox, area = yolo_obb_to_coco(yolo_obb_data, img_width, img_height)

            reason = ""
            if bbox[2] <= 0 or bbox[3] <= 0 or area <= 1:
                skipped_annotations_count += 1
                print(f"\nFile: {label_file.name}")
                print(f"Line in File: {line_num}")
                print(f"Reason: {reason}")
                print(f"Bbox [x,y,w,h]: {bbox}")
                print(f"Area: {area}")
                print(f"Segmentation: {segmentation}")
                continue

            annotation_info = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": class_id,
                "bbox": bbox,
                "segmentation": [segmentation],
                "area": area,
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation_info)
            annotation_id_counter += 1

        image_id_counter += 1

    print(f"\nProcessed {image_id_counter - 1} images and {annotation_id_counter - 1} annotations.")
    print(f"Skipped {skipped_annotations_count} annotations.")

    ouput_json_path = root_path / "master_annotations.json"
    with open(ouput_json_path, "w") as f:
        json.dump(coco_data, f, indent=4)


def coco_to_yolo_obb(split_dir: str):
    split_path = Path(split_dir)
    json_path = split_path / "sliced_annotations.json_coco.json"
    labels_path = split_path / "labels"

    labels_path.mkdir(parents=True, exist_ok=True)
    with open(json_path) as f:
        coco_data = json.load(f)

    annotations_by_image_id = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image_id[ann["image_id"]].append(ann)

    files_with_annotations = 0
    empty_files_created = 0

    for image_info in tqdm(coco_data["images"], desc=f"Creating YOLO files for '{split_path.name}'"):
        image_id = image_info["id"]

        label_filename = Path(image_info["file_name"]).stem + ".txt"
        output_path = labels_path / label_filename

        yolo_lines = []

        if image_id in annotations_by_image_id:
            img_width = image_info["width"]
            img_height = image_info["height"]

            for ann in annotations_by_image_id[image_id]:
                class_id = ann["category_id"]
                abs_coords = ann["segmentation"][0]

                if len(abs_coords) != 8:
                    continue

                normalized_coords = []
                for i, coord in enumerate(abs_coords):
                    if i % 2 == 0:
                        normalized_coords.append(coord / img_width)
                    else:
                        normalized_coords.append(coord / img_height)

                yolo_coords_str = " ".join([f"{c:.6f}" for c in normalized_coords])
                yolo_lines.append(f"{class_id} {yolo_coords_str}")

            if yolo_lines:
                files_with_annotations += 1
            else:
                empty_files_created += 1
        else:
            empty_files_created += 1

        with open(output_path, "w") as f:
            f.write("\n".join(yolo_lines))

    total_images = len(coco_data["images"])
    print(f"\n--- Conversion Summary for '{split_path.name}' ---")
    print(f"Total images processed: {total_images}")
    print(f"Label files with annotations: {files_with_annotations}")
    print(f"Empty label files created (no annotations): {empty_files_created}")
    print(f"All label files saved to: {labels_path}")


def create_label_files_from_master_json(root_dir: str):
    sliced_root_path = Path(root_dir)

    splits_to_process = [d for d in sliced_root_path.iterdir() if d.is_dir()]

    for split_dir in splits_to_process:
        print("\n" + "=" * 60)
        print(f"Processing split: {split_dir.name}")
        print("=" * 60)
        coco_to_yolo_obb(str(split_dir))


if __name__ == "__main__":
    # create_master_coco_json("D:\\stuff\\datasets\\MSGOv1")
    create_label_files_from_master_json("D:\\stuff\\datasets\\MSGOv1\\sliced")
