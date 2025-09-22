from ultralytics import YOLO

DATASET_YAML = "../../dataset/MSGOv2_small.yaml"


def train_yolo_model():
    MODEL_NAME = "../../pretrained_models/yolo11n.pt"

    EPOCHS = 3
    BATCH_SIZE = 16
    IMAGE_SIZE = 800
    PATIENCE = 10

    model = YOLO(MODEL_NAME)

    _ = model.train(data=DATASET_YAML, imgsz=IMAGE_SIZE, epochs=EPOCHS, batch=BATCH_SIZE, patience=PATIENCE)


def test_yolo_model():
    model = YOLO("runs/detect/train3/weights/best.pt")

    _ = model.val(data=DATASET_YAML, split="test", name="test")


if __name__ == "__main__":
    train_yolo_model()
    test_yolo_model()
