import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def convert_png_to_jpg(png_path, jpg_path):
    img = Image.open(png_path)
    img.save(jpg_path)


def walkdir_and_convert(directory):
    for file in os.listdir(directory):
        if file.endswith(".png"):
            png_path = os.path.join(directory, file)
            jpg_path = os.path.splitext(png_path)[0] + ".jpg"

            convert_png_to_jpg(png_path, jpg_path)

            os.remove(png_path)


def main():
    folders = ["D:\\stuff\\datasets\\DOTAv2\\DOTAv2\\images\\val", "D:\\stuff\\datasets\\DOTAv2\\DOTAv2\\images\\train"]
    for folder in folders:
        walkdir_and_convert(folder)


if __name__ == "__main__":
    main()
