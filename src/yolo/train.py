from ultralytics import YOLO

DATASET_YAML = "MSGOv1.yaml"


def train_yolo_model():
    MODEL_NAME = "yolo11n-obb.pt"

    EPOCHS = 100
    BATCH_SIZE = 16
    IMAGE_SIZE = 800
    PATIENCE = 10

    model = YOLO(MODEL_NAME)

    print("Start model training ...")

    _ = model.train(data=DATASET_YAML, imgsz=IMAGE_SIZE, epochs=EPOCHS, batch=BATCH_SIZE, patience=PATIENCE)

    print("Training complete.")


def test_yolo_model():
    model = YOLO("runs/obb/train/weights/best.pt")

    _ = model.val(data=DATASET_YAML, split="test", name="test")


if __name__ == "__main__":
    train_yolo_model()
    test_yolo_model()
