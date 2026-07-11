import gc
import weakref

import matplotlib
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRNano  # RFDETRSmall, RFDETRMedium, RFDETRLarge
from supervision.metrics import MeanAveragePrecision
from tqdm import tqdm

matplotlib.use("Agg")

DATASET_LOCATION = "../../dataset/MSGOv1"


def cleanup_gpu_memory(obj=None, verbose: bool = False):
    if not torch.cuda.is_available():
        if verbose:
            print("[INFO] CUDA is not available. No GPU cleanup needed.")
        return

    def get_memory_stats():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved

    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(f"[Before] Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")

    # Ensure we drop all strong references
    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            print("[WARNING] Object not fully garbage collected yet.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(f"[After]  Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")


def train_rfdetr_model():
    EPOCHS = 6
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8
    EARLY_STOPPING = False
    PATIENCE = 10

    cuda = torch.cuda.is_available()

    if not cuda:
        print("GPU not avaiable, training not possible..")
        return

    model = RFDETRNano()

    # batch_size * grad_accum_steps = 16
    model.train(
        dataset_dir=DATASET_LOCATION,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        early_stopping=EARLY_STOPPING,
        early_stopping_patience=PATIENCE,
    )

    cleanup_gpu_memory(model, verbose=True)


def test_rfdetr_model():
    model = RFDETRNano(pretrain_weights="output/checkpoint_best_total.pth")
    model.optimize_for_inference()

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=f"{DATASET_LOCATION}/test",
        annotations_path=f"{DATASET_LOCATION}/test/_annotations.coco.json",
    )

    targets = []
    predictions = []

    for path, _, annotations in tqdm(ds):
        image = Image.open(path)
        detections = model.predict(image, threshold=0)

        targets.append(annotations)
        predictions.append(detections)

    map_metric = MeanAveragePrecision()
    map_result = map_metric.update(predictions, targets).compute()

    print(map_result)


if __name__ == "__main__":
    train_rfdetr_model()
    test_rfdetr_model()
