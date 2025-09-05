import json


def convert_filenames_json(input_json_path, output_json_path, from_ext=".bmp", to_ext=".jpg"):
    with open(input_json_path) as f:
        data = json.load(f)

    changes_count = 0
    if "images" in data:
        for image_info in data["images"]:
            if "file_name" in image_info and image_info["file_name"].endswith(from_ext):
                original_name = image_info["file_name"]
                image_info["file_name"] = original_name.replace(from_ext, to_ext)
                changes_count += 1

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("Successfully processed file.")
    print(f"Changed {changes_count} filenames from '{from_ext}' to '{to_ext}'.")
    print(f"Output saved to: {output_json_path}")


def main():
    train_json_original = "D:\\stuff\\datasets\\ShipRSImageNet\\annotations\\ShipRSImageNet_bbox_train_level_1.json"
    val_json_original = "D:\\stuff\\datasets\\ShipRSImageNet\\annotations\\ShipRSImageNet_bbox_val_level_1.json"

    train_json_converted = "D:\\stuff\\datasets\\ShipRSImageNet\\annotations\\ShipRSImageNet_bbox_train_level_1jpg.json"
    val_json_converted = "D:\\stuff\\datasets\\ShipRSImageNet\\annotations\\ShipRSImageNet_bbox_val_level_1jpg.json"

    convert_filenames_json(train_json_original, train_json_converted)
    convert_filenames_json(val_json_original, val_json_converted)


if __name__ == "__main__":
    main()
