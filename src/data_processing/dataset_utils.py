import random
from pathlib import Path

from tqdm import tqdm


def get_empty_image_paths(root_path: Path) -> list[Path]:
    results = []

    images_dir = root_path / "images"
    labels_dir = root_path / "labels"

    for img_file in images_dir.iterdir():
        img_name = img_file.stem
        label_file = labels_dir / f"{img_name}.txt"

        if label_file.stat().st_size == 0:
            results.append(img_file)

    return results


def delete_empty_images(root_dir: str) -> None:
    root_path = Path(root_dir)

    labels_dir = root_path / "labels"

    empty_image_paths = get_empty_image_paths(root_path)

    for img_file in tqdm(empty_image_paths, desc="Deleting empty images and labels..."):
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


def delete_some_empty_images(root_dir: str, keep_percentage: float = 0.15) -> None:
    root_path = Path(root_dir)

    for split_path in root_path.iterdir():
        if not split_path.is_dir():
            continue

        labels_dir = split_path / "labels"

        empty_image_paths = get_empty_image_paths(split_path)
        num_empty = len(empty_image_paths)
        num_to_keep = round(num_empty * keep_percentage)

        if num_to_keep >= num_empty:
            print(f"[{split_path.name}] Found {num_empty} empty images. Keeping all.")
            continue

        num_to_delete = num_empty - num_to_keep
        images_to_delete = random.sample(empty_image_paths, num_to_delete)

        for img_file in tqdm(images_to_delete, desc=f"Deleting empty images and labels from {split_path.name}"):
            label_file = labels_dir / f"{img_file.stem}.txt"

            try:
                img_file.unlink()
                label_file.unlink()
            except Exception as e:
                print(f"Failed to delete image {img_file.name}: {e}")


def rename_files(root_dir: str) -> None:
    root_path = Path(root_dir)

    for dataset_path in root_path.iterdir():
        if not dataset_path.is_dir():
            continue

        dataset_name = dataset_path.stem

        if dataset_name == "DIOR":
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"

            label_files = list(labels_dir.glob("*.txt"))
            for label_file in tqdm(label_files, desc=f"Renaming labels from {dataset_name}..."):
                new_label_file_name = f"{dataset_name}_{label_file.name}"
                new_label_path = labels_dir / new_label_file_name
                label_file.rename(new_label_path)

            img_files = list(images_dir.glob("*.jpg"))
            for img_file in tqdm(img_files, desc=f"Renaming images from {dataset_name}..."):
                new_img_file_name = f"{dataset_name}_{img_file.name}"
                new_img_path = images_dir / new_img_file_name
                img_file.rename(new_img_path)

        else:
            for split_path in dataset_path.iterdir():
                if not split_path.is_dir():
                    continue

                images_dir = split_path / "images"
                labels_dir = split_path / "labels"

                dataset_name = split_path.parent.name
                split_name = split_path.name

                label_files = list(labels_dir.glob("*.txt"))
                for label_file in tqdm(label_files, desc=f"Renaming labels from {dataset_name} {split_name}..."):
                    new_label_file_name = f"{dataset_name}_{split_name}_{label_file.name}"
                    new_label_path = labels_dir / new_label_file_name
                    label_file.rename(new_label_path)

                img_files = list(images_dir.glob("*.jpg"))
                for img_file in tqdm(img_files, desc=f"Renaming images from {dataset_name} {split_name}..."):
                    new_img_file_name = f"{dataset_name}_{split_name}_{img_file.name}"
                    new_img_path = images_dir / new_img_file_name
                    img_file.rename(new_img_path)


def move_png_files(source_folder: str) -> None:
    source_path = Path(source_folder)
    dest_path = source_path / "images"
    dest_path.mkdir(parents=True, exist_ok=True)

    png_files = list(source_path.glob("*.png"))
    for png_file in tqdm(png_files, desc="Moving files..."):
        png_file.rename(dest_path / png_file.name)


if __name__ == "__main__":
    # rename_files("D:\\stuff\\datasets\\MSGOv2")
    # move_png_files("D:\\stuff\\datasets\\MSGOv2\\sliced\\test")
    # move_png_files("D:\\stuff\\datasets\\MSGOv2\\sliced\\train")
    # delete_empty_images("D:\\stuff\\datasets\\MSGOv2")
    delete_some_empty_images("D:\\stuff\\datasets\\MSGOv2\\sliced")
