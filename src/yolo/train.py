from PIL import Image
from ultralytics import YOLO

DATASET_YAML = "../../dataset/MSGOv1.yaml"


def train_yolo_model():
    MODEL_NAME = "../../pretrained_models/yolo12n.pt"

    EPOCHS = 100
    BATCH_SIZE = 8
    IMAGE_SIZE = 800
    PATIENCE = 10

    model = YOLO(MODEL_NAME)

    _ = model.train(data=DATASET_YAML, imgsz=IMAGE_SIZE, epochs=EPOCHS, batch=BATCH_SIZE, patience=PATIENCE)


def test_yolo_model():
    model = YOLO("runs/detect/train8/weights/best.pt")

    _ = model.val(data=DATASET_YAML, split="test", name="test")


def predict():
    model = YOLO("../../models/yolo12m_best.pt")

    results = model("D:\\stuff\\datasets\\DIOR\\archive\\images\\00008.jpg")

    for _, r in enumerate(results):
        im_bgr = r.plot(line_width=1, labels=False, conf=False)  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        im_rgb.show()


if __name__ == "__main__":
    train_yolo_model()
    # test_yolo_model()
    # predict()
