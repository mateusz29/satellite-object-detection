import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def convert_to_jpg(image_path, jpg_path):
    img = Image.open(image_path)
    img.save(jpg_path)


def walkdir_and_convert(directory):
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".bmp"):
            image_path = os.path.join(directory, file)
            jpg_path = os.path.splitext(image_path)[0] + ".jpg"

            convert_to_jpg(image_path, jpg_path)

            os.remove(image_path)


def main():
    folders = ["D:\\stuff\\datasets\\ShipRSImageNet\\images"]
    for folder in folders:
        walkdir_and_convert(folder)


if __name__ == "__main__":
    main()
