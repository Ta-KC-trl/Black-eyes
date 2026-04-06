from ultralytics import YOLO

model = YOLO("yolov8m.pt")

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