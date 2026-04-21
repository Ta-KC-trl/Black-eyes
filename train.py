from ultralytics import YOLO
import os
import yaml

_BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE, "config.yaml"), "r") as _f:
    _cfg = yaml.safe_load(_f)

if __name__ == "__main__":
    data_yaml = os.path.join(_BASE, _cfg["PATH"].get("DATASET_DIR", "dataset/"), "data.yaml")
    model = YOLO(_cfg["YOLO"].get("BASE_MODEL", "yolo11n.pt"))

    model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        name="knife_openimages_v1"
    )

    print("Done! Model at: runs/detect/knife_openimages_v1/weights/best.pt")
