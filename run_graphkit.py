import os
import subprocess
import sys

def run_gridfm_training():
    # Define the path to the example config included in the repo
    # Note: Ensure this path matches the 'examples' folder in your cloned repo
    config_path = os.path.join("examples", "config", "case30_ieee_base.yaml")
    data_path = os.path.join("examples", "data")

    # Check if the example config exists
    if not os.path.exists(config_path):
        print(f"Error: Could not find example config at {config_path}")
        print("Make sure you are running this script from the root of the 'gridfm-graphkit' repository.")
        return

    print(f"🚀 Starting GridFM training using config: {config_path}")
    
    # Construct the command
    # Equivalent to: gridfm_graphkit train --config examples/config/... --data_path examples/data
    command = [
        "gridfm_graphkit",
        "train",
        "--config", config_path,
        "--data_path", data_path
    ]

    try:
        # Run the command and stream output to the console
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}.")
        print("Ensure you have installed the package with: pip install -e .")
    except FileNotFoundError:
        print("\n❌ 'gridfm_graphkit' command not found.")
        print("Did you activate your virtual environment?")

if __name__ == "__main__":
    run_gridfm_training()