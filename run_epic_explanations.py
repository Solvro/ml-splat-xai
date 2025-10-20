import subprocess
import os

ply_files = [
    "toys_ds_cleaned/test/2/cup_001.ply",
    "toys_ds_cleaned/test/3/deer_moose_016.ply",
    "toys_ds_cleaned/test/4/dinosaur_001.ply",
    "toys_ds_cleaned/test/5/fox_005.ply",
    "toys_ds_cleaned/test/6/glass_012.ply",
    "toys_ds_cleaned/test/7/hat_000.ply",
    "toys_ds_cleaned/test/8/monitor_005.ply",
    "toys_ds_cleaned/test/9/sofa_000.ply",
    "toys_ds_cleaned/test/10/tree_001.ply"
]

epic_script = "./explain_epic.py"  # Adjust if the script is in a different location

def run_epic_explanation(ply_path):
    try:
        # Run the LIME_single script with the specified PLY file
        command = [".venv/Scripts/python.exe", epic_script, "--ply_path", ply_path]
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully processed: {ply_path}")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {ply_path}")
        print(f"Error message: {e.stderr}")

def main():
    # Process each PLY file in the list
    for ply_file in ply_files:
        if os.path.exists(ply_file):
            print(f"\nProcessing: {ply_file}")
            run_epic_explanation(ply_file)
        else:
            print(f"File not found: {ply_file}")

if __name__ == "__main__":
    main()