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


if __name__ == "__main__":
    delete_some_empty_images("D:\\stuff\\datasets\\MSGOv1\\YOLODIOR-R")
