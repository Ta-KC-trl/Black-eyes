from ultralytics import YOLO
import os, yaml

_BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE, "config.yaml"), "r") as _f:
    _cfg = yaml.safe_load(_f)

model = YOLO(_cfg["YOLO"].get("BASE_MODEL", "yolov8m.pt"))

results = model.train(
    data="data/knife_openimages/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    name="knife_openimages_v1"
)

print("Done! Model at: runs/detect/knife_openimages_v1/weights/best.pt")