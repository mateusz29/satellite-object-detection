from pathlib import Path

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def to_jpg(image_path: Path) -> None:
    file_name = image_path.stem + ".jpg"
    jpg_path = image_path.parent / file_name

    img = Image.open(image_path)
    rgb_img = img.convert("RGB")
    rgb_img.save(jpg_path)


def convert_images(root_dir: str) -> None:
    root_path = Path(root_dir)

    for split_path in root_path.iterdir():
        images_dir = split_path / "images"

        img_files = list(images_dir.iterdir())
        for img_file in tqdm(img_files, desc=f"Converting images in {split_path}"):
            if img_file.suffix == ".png":
                to_jpg(img_file)

                try:
                    img_file.unlink()
                except Exception as e:
                    print(f"Failed to delete image {img_file}: {e}")


def main():
    convert_images("D:\\stuff\\datasets\\MSGOv2\\MSGOv2\\sliced")


if __name__ == "__main__":
    main()
