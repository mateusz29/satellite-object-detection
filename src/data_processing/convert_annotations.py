from pathlib import Path

from PIL import Image

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


if __name__ == "__main__":
    # walkdir_fair1m_and_convert("D:\\stuff\\datasets\\MSGOv1\\FAIR1M")
    # walkdir_dotav2_and_convert("D:\\stuff\\datasets\\MSGOv1\\DOTAv2")
    walkdir_dior_and_convert("D:\\stuff\\datasets\\MSGOv1\\YOLODIOR-R")
