import os
import shutil

files_to_delete = [
    "Tracking.py", "utils.py", "config.yaml", "11.pt", "best.pt", "new.pt",
    "yolov8x.pt", "yolov8m.pt", "yolov8s.pt", "yolo26n.pt", "knife.yolov8.zip",
    "fix.py", "reorganize.py", "merge_datasets.py", "fix_dataset.py",
    "download_openimages.py", "tmp_check_models.py", "model_names.txt",
    "train.py", "model_check.txt"
]

for f in files_to_delete:
    if os.path.exists(f):
        try:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Error deleting {f}: {e}")
    else:
        print(f"Skipped: {f} (not found)")
