from ultralytics import YOLO
import argparse
import os
import yaml

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cfg_path = os.path.join(_BASE, "config.yaml")
with open(_cfg_path, "r") as _f:
    _cfg = yaml.safe_load(_f)
_DEFAULT_BASE_MODEL = _cfg["YOLO"].get("BASE_MODEL", "yolo11n.pt")

def train_knife_detection(data_yaml_path, epochs=100, model_type=None):
    if model_type is None:
        model_type = _DEFAULT_BASE_MODEL
    print(f"--- Starting Training with YOLOv11 ---")
    model = YOLO(model_type)
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0, # Use GPU if available
        workers=4,
        name="knife_v11_professional"
    )
    
    print(f"Training Complete! Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Anomaly Detection")
    parser.add_argument("--data", type=str, default="data/knife_openimages/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model", type=str, default=_DEFAULT_BASE_MODEL, help="Base model to use (yolo11n.pt, yolo11s.pt, etc.)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.data):
        train_knife_detection(args.data, args.epochs, args.model)
    else:
        print(f"Error: Dataset YAML not found at {args.data}")
