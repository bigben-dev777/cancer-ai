import glob
import os
import subprocess

# Folder where your models are stored
model_folder = "manager/models/2025-11-27"
class_list = ["AKIEC", "BCC", "BKL", "DF", "INF", "MEL", "NV", "SCCKA"]

for class_item in class_list:
    # Dataset directory
    dataset_dir = f"manager/dataset/csv_class/{class_item}"

    # Competition ID
    competition_id = "tricorder-3"

    # Find all .onnx files recursively in subfolders
    model_paths = glob.glob(os.path.join(model_folder, "**", "*.onnx"), recursive=True)

    # Loop over each model and call the evaluation script
    for model_path in model_paths:
        cmd = [
            "python",
            "manager/scripts/local_miner.py",
            "--action",
            "evaluate",
            "--competition_id",
            competition_id,
            "--model_path",
            model_path,
            "--dataset_dir",
            dataset_dir,
        ]
        print(f"Evaluating model: {model_path}")
        print(cmd)
        subprocess.run(cmd)
