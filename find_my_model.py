import os

def find_latest_checkpoint(root_dir="mlruns"):
    latest_time = 0
    latest_file = None

    # Walk through all folders in mlruns to find .ckpt or .pth files
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ckpt") or file.endswith(".pth"):
                filepath = os.path.join(subdir, file)
                file_time = os.path.getmtime(filepath)
                # Keep track of the newest file
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = filepath

    return latest_file

model_path = find_latest_checkpoint()

if model_path:
    print("\nSUCCESS! Found your latest model here:")
    print(f"{model_path}")
    print("\nCopy the path above for your command.")
else:
    print("Could not find any model files in 'mlruns'.")