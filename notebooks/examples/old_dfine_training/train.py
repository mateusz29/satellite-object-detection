from pprint import pprint

import albumentations as A
import torch
from coco_dataset_loader import CocoDatasetLoader
from datasets import DatasetDict
from map_evaluator import MAPEvaluator
from msgo_dataset import MSGODataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DATASET_LOCATION = "../../dataset/MSGOv1"
MODEL_NAME = "ustc-community/dfine-medium-coco"
IMAGE_SIZE = 800
EPOCHS = 100
BATCH_SIZE = 4


def load_dataset() -> DatasetDict:
    coco_loader = CocoDatasetLoader(DATASET_LOCATION)

    dataset = DatasetDict()

    for split in ["train", "valid", "test"]:
        hf_dataset = coco_loader.load_coco_hf_dataset(split)
        if len(hf_dataset) > 0:
            dataset[split] = hf_dataset
            print(hf_dataset)

    return dataset


def preprocess_dataset():
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        do_resize=True,
        size={"width": IMAGE_SIZE, "height": IMAGE_SIZE},
        use_fast=True,
    )

    train_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=25, min_width=1, min_height=1
        ),
    )

    # to make sure boxes are clipped to image size and there is no boxes with area < 1 pixel
    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1
        ),
    )

    return image_processor, train_transform, validation_transform


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


def train_dfine_model():
    dataset = load_dataset()
    image_processor, train_transform, validation_transform = preprocess_dataset()

    train_dataset = MSGODataset(dataset["train"], image_processor, transform=train_transform)
    validation_dataset = MSGODataset(dataset["valid"], image_processor, transform=validation_transform)
    test_dataset = MSGODataset(dataset["test"], image_processor, transform=validation_transform)

    label2id = {
        "Plane": 0,
        "Bridge": 1,
        "Airport": 2,
        "Harbor": 3,
        "Vehicle": 4,
        "Ship": 5,
    }
    id2label = {v: k for k, v in label2id.items()}

    eval_compute_metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=0.01, id2label=id2label)

    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="d-fine-n-msgo-finetune-1",
        num_train_epochs=EPOCHS,
        max_grad_norm=0.1,
        learning_rate=5e-5,
        warmup_steps=300,
        per_device_train_batch_size=BATCH_SIZE,
        dataloader_num_workers=0,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="tensorboard",  # or "wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    )

    trainer.train()

    # Test the trained model
    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval")
    pprint(metrics)


if __name__ == "__main__":
    train_dfine_model()
