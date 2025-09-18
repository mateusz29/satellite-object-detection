from pathlib import Path


def get_empty_image_paths(root_path: Path) -> list[Path]:
    results = []

    for split_path in root_path.iterdir():
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        for img_file in images_dir.iterdir():
            img_name = img_file.stem
            label_file = labels_dir / f"{img_name}.txt"

            if label_file.stat().st_size == 0:
                results.append(img_file)

    return results


def delete_empty_images(root_dir: str) -> None:
    root_path = Path(root_dir)
    empty_image_paths = get_empty_image_paths(root_path)

    for img_file in empty_image_paths:
        labels_dir = img_file.parent.parent / "labels"
        label_file = labels_dir / f"{img_file.stem}.txt"

        try:
            img_file.unlink()
        except Exception as e:
            print(f"Failed to delete image {img_file}: {e}")

        if label_file.exists():
            try:
                label_file.unlink()
            except Exception as e:
                print(f"Failed to delete label {label_file}: {e}")


def rename_files(root_dir: str) -> None:
    root_path = Path(root_dir)

    for dataset_path in root_path.iterdir():
        if not dataset_path.is_dir():
            continue

        for split_path in dataset_path.iterdir():
            if not split_path.is_dir():
                continue

            images_dir = split_path / "images"
            labels_dir = split_path / "labels"

            dataset_name = split_path.parent.name
            split_name = split_path.name

            for label_file in labels_dir.glob("*.txt"):
                new_label_file_name = f"{dataset_name}_{split_name}_{label_file.name}"
                new_label_path = labels_dir / new_label_file_name
                label_file.rename(new_label_path)

                print(f"Renamed {label_file} -> {new_label_path}")

            for img_file in images_dir.glob("*.jpg"):
                new_img_file_name = f"{dataset_name}_{split_name}_{img_file.name}"
                new_img_path = images_dir / new_img_file_name
                img_file.rename(new_img_path)

                print(f"Renamed {img_file} -> {new_img_path}")


def move_png_files(source_folder: str) -> None:
    source_path = Path(source_folder)
    dest_path = source_path / "images"
    dest_path.mkdir(parents=True, exist_ok=True)

    for png_file in source_path.glob("*.png"):
        png_file.rename(dest_path / png_file.name)


if __name__ == "__main__":
    # rename_files("D:\\stuff\\datasets\\MSGOv1")
    # move_png_files("D:\\stuff\\datasets\\MSGOv1\\sliced\\val")
    delete_empty_images("D:\\stuff\\datasets\\MSGOv1\\sliced")
