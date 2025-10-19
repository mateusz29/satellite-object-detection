import json
import os

from datasets import ClassLabel, Dataset, Features, Image, Sequence, Value


class CocoDatasetLoader:
    def __init__(self, coco_folder):
        self.coco_folder = coco_folder

    def group_by_key_id(self, data, key_id, category_id_to_index):
        """
        Groups data by a specified key and maps category IDs to indices.

        Args:
            data (list): List of dictionaries containing the data.
            key_id (str): The key to group by.
            category_id_to_index (dict): Mapping from category IDs to indices.

        Returns:
            dict: Grouped data.
        """
        grouped_data = {}
        for item in data:
            key_value = item[key_id]
            if key_value not in grouped_data:
                grouped_data[key_value] = {k: [] for k in item if k != key_id}
            for k, v in item.items():
                if k != key_id:
                    grouped_data[key_value][k].append(v)
            grouped_data[key_value]["category"] = [
                category_id_to_index[x] for x in grouped_data[key_value]["category_id"]
            ]
        return grouped_data

    def load_coco_hf_dataset(self, split):
        """
        Loads COCO dataset and processes it into a format suitable for Hugging Face datasets.

        Args:
            split (str): Dataset split (e.g., 'Train', 'Test', 'Validation').

        Returns:
            Dataset: HuggingFace Dataset of the split of COCO dataset.
        """
        # Load the JSON file
        json_file_path = os.path.join(self.coco_folder, split, "_annotations.coco.json")

        try:
            with open(json_file_path) as f:
                coco_data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {json_file_path}")
            return []

        # Extract category names and create a mapping from category IDs to indices
        category_names = [cat["name"] for cat in coco_data["categories"]]
        category_id_to_index = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

        # Group annotations by 'image_id'
        grouped_annotations = self.group_by_key_id(coco_data["annotations"], "image_id", category_id_to_index)

        # Create a dictionary of images
        grouped_images = {item["id"]: item for item in coco_data["images"]}

        # Initialize 'objects' field in grouped_images
        annotations_keys = next(iter(grouped_annotations.values())).keys()
        for k, _ in grouped_images.items():
            grouped_images[k]["objects"] = {key: [] for key in annotations_keys}

        # Populate 'objects' field with annotations
        for k, v in grouped_annotations.items():
            grouped_images[k]["objects"] = v

        # Add image paths and IDs
        for _, v in grouped_images.items():
            v["image"] = os.path.join(self.coco_folder, split, v["file_name"])
            v["image_id"] = v["id"]

        # Create a Hugging Face dataset from the custom data using from_list for efficiency
        hf_dataset = Dataset.from_list(list(grouped_images.values()))

        # Define the features for the main dataset
        features = Features(
            {
                "id": Value("int64"),
                "image_id": Value("int64"),
                "image": Image(),
                "file_name": Value("string"),
                "width": Value("int64"),
                "height": Value("int64"),
                "objects": Sequence(
                    {
                        "id": Value("int64"),
                        "area": Value("float32"),
                        "bbox": Sequence(Value("float32")),
                        "category": ClassLabel(names=category_names),
                        "attributes": {"occluded": Value("bool")},
                        "category_id": Value("int64"),
                        "iscrowd": Value("int64"),
                        "segmentation": Sequence(Sequence(Value("float32"))),
                    }
                ),
            }
        )

        # Cast the features for the Hugging Face dataset
        hf_dataset = hf_dataset.cast(features)

        return hf_dataset
