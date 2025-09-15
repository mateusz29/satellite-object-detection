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

    for split_path in root_path.iterdir():
        if not split_path.is_dir() or split_path.is_file():
            continue

        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        for label_file in labels_dir.glob("*.txt"):
            img_file = images_dir / f"{label_file.stem}.jpg"

            counts = {}
            with open(label_file) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                counts[class_id] = counts.get(class_id, 0) + 1

            image_to_counts[str(img_file)] = counts

    return image_to_counts


def rename_files(root_dir: str) -> None:
    root_path = Path(root_dir)

    for dataset_path in root_path.iterdir():
        for split_path in dataset_path.iterdir():
            if not split_path.is_dir() or split_path.stem == "labels":
                continue

            images_dir = split_path / "images"
            labels_dir = split_path / "labels"

            dataset_name = split_path.parent.name

            for label_file in labels_dir.glob("*.txt"):
                if label_file.name.startswith(dataset_name):
                    continue

                new_label_file_name = f"{dataset_name}_{label_file.name}"
                new_label_path = labels_dir / new_label_file_name
                label_file.rename(new_label_path)

                print(f"Renamed {label_file} -> {new_label_path}")

            for img_file in images_dir.glob("*.jpg"):
                if img_file.name.startswith(dataset_name):
                    continue

                new_img_file_name = f"{dataset_name}_{img_file.name}"
                new_img_path = images_dir / new_img_file_name
                img_file.rename(new_img_path)

                print(f"Renamed {img_file} -> {new_img_path}")


if __name__ == "__main__":
    rename_files("D:\\stuff\\datasets\\MSGOv1")
