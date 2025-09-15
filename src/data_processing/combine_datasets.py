import random
from pathlib import Path


def get_empty_image_paths(root_path: Path) -> list[Path]:
    results = []

    for split_path in root_path.iterdir():
        if not split_path.is_dir():
            continue

        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        for img_file in images_dir.iterdir():
            img_name = img_file.stem
            label_file = labels_dir / f"{img_name}.txt"

            if label_file.stat().st_size == 0:
                results.append(img_file)

    return results


def delete_some_empty_images(root_dir: str) -> None:
    target_empty_count = 900

    root_path = Path(root_dir)
    empty_image_paths = get_empty_image_paths(root_path)

    if len(empty_image_paths) > target_empty_count:
        sampled_empty_paths = random.sample(empty_image_paths, target_empty_count)
    else:
        sampled_empty_paths = empty_image_paths

    for img_file in empty_image_paths:
        if img_file in sampled_empty_paths:
            continue

        labels_dir = img_file.parent.parent / "labels"
        label_file = labels_dir / f"{img_file.stem}.txt"

        try:
            img_file.unlink()
            print(f"Deleted an image: {img_file}")
        except Exception as e:
            print(f"Failed to delete image {img_file}: {e}")

        if label_file.exists():
            try:
                label_file.unlink()
                print(f"Deleted a label: {label_file}")
            except Exception as e:
                print(f"Failed to delete label {label_file}: {e}")


def build_image_to_counts(root_dir: str) -> dict[str, dict[int, int]]:
    root_path = Path(root_dir)
    image_to_counts = {}

    for dataset_path in root_path.iterdir():
        for split_path in dataset_path.iterdir():
            if not split_path.is_dir() or split_path.stem == "labels":
                continue

            images_dir = split_path / "images"
            labels_dir = split_path / "labels"

            for label_file in labels_dir.glob("*.txt"):
                img_file = images_dir / f"{label_file.stem}.jpg"

                counts = {}
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_id = int(parts[0])
                        counts[class_id] = counts.get(class_id, 0) + 1

                if counts:
                    image_to_counts[str(img_file)] = counts

    return image_to_counts


if __name__ == "__main__":
    # delete_some_empty_images("D:\\stuff\\datasets\\MSGOv1\\YOLODIOR-R")
    image_to_counts = build_image_to_counts("D:\\stuff\\datasets\\MSGOv1")
    print(image_to_counts)
