import os
import shutil
import glob

def reorganize():
    # Define directories
    dirs = ['src', 'assets', 'scripts', 'data/face_db']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # Files to move to src/
    for f in ['Tracking.py', 'utils.py']:
        if os.path.exists(f):
            shutil.move(f, os.path.join('src', f))
            print(f"Moved {f} to src/")

    # Files to move to data/
    if os.path.exists('config.yaml'):
        shutil.move('config.yaml', os.path.join('data', 'config.yaml'))
        print("Moved config.yaml to data/")

    # Files to move to scripts/
    script_files = [
        'download_openimages.py', 'fix.py', 'fix_dataset.py', 
        'merge_datasets.py', 'train.py'
    ]
    for f in script_files:
        if os.path.exists(f):
            shutil.move(f, os.path.join('scripts', f))
            print(f"Moved {f} to scripts/")

    # Move .pt models to models/
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    pt_files = glob.glob('*.pt')
    for f in pt_files:
        dest = os.path.join(models_dir, f)
        if os.path.exists(dest):
            os.remove(dest) # Avoid collision if already there
        shutil.move(f, dest)
        print(f"Moved {f} to models/")

    # Move assets
    asset_dirs = ['Sample_images', 'Background']
    for d in asset_dirs:
        if os.path.exists(d):
            dest = os.path.join('assets', d)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(d, dest)
            print(f"Moved {d} to assets/")

    # Cleanup unwanted files/folders
    unwanted = ['dataset_v2', 'knife.yolov8.zip', 'knife.yolov8']
    for item in unwanted:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
            print(f"Removed unwanted item: {item}")

if __name__ == "__main__":
    reorganize()
