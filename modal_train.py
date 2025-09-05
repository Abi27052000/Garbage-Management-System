# modal_train.py
import modal
from ultralytics import YOLO

app = modal.App("Garbage-Detection")

dataset_vol = modal.Volume.from_name("garbage-dataset")
model_vol = modal.Volume.from_name("garbage-models", create_if_missing=True)


image = modal.Image.from_registry("ultralytics/ultralytics:latest")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/root/dataset": dataset_vol, "/root/output": model_vol},
    timeout=60 * 60 * 6,
)
def train_garbage_detection_model():
    # model = YOLO("yolo11n.pt")
    # model = YOLO("yolo11n.pt")
    # model = YOLO("yolo11s.pt")
    model = YOLO("yolo11m.pt")

    model.train(
        data="/root/dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="/root/output",
        name="garbage-detect",
    )
