import os
import json
import re

IGNORE_FAILED_TRAJECTORIES = False

def find_intervention_segments(trajectory_data):
    """
    Identify contiguous segments where intervention_occuring is True.
    Each trajectory record is assumed to be a list in the format:
      [observation, action, intervention_occuring, image_file]
    Returns a list of tuples: (start_index, end_index) where end_index is exclusive.
    """
    segments = []
    i = 0
    n = len(trajectory_data)
    while i < n:
        # Check if the third element exists and is True
        if len(trajectory_data[i]) > 2 and trajectory_data[i][2] is True:
            start = i
            while i < n and len(trajectory_data[i]) > 2 and trajectory_data[i][2] is True:
                i += 1
            end = i
            # Consider segments longer than a minimum length (e.g., 2 steps)
            if (end - start) > 2:
                segments.append((start, end))
        else:
            i += 1
    return segments

def count_interventions_in_file(trajectory_file):
    """
    Load a trajectory file and count the number of intervention segments.
    Returns the number of segments, or None if the file is skipped.
    """
    try:
        with open(trajectory_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {trajectory_file}. Skipping.")
        return None
    except Exception as e:
        print(f"Error loading {trajectory_file}: {e}. Skipping.")
        return None

    trajectory_data = data.get("trajectory", [])
    if not trajectory_data:
        # print(f"No trajectory data found in {trajectory_file}")
        return 0 # Count as 0 interventions if no trajectory data

    if IGNORE_FAILED_TRAJECTORIES and data.get("metrics", {}).get("success_rate") == 0:
        # print(f"Skipping {os.path.basename(trajectory_file)}: failed trajectory.")
        return None # Indicate that this file should not be counted towards the average

    segments = find_intervention_segments(trajectory_data)
    return len(segments)

def process_directory(trajectory_dir):
    """
    Process all trajectory files in the specified directory and calculate the average number of interventions.
    Only files starting with 'trajectory_' and ending with '.json' will be processed.
    """
    total_interventions = 0
    processed_files_count = 0
    skipped_files_count = 0

    filenames = sorted(os.listdir(trajectory_dir)) # Sort for consistent processing order

    for filename in filenames:
        if filename.startswith("trajectory_") and filename.endswith(".json"):
            filepath = os.path.join(trajectory_dir, filename)
            intervention_count = count_interventions_in_file(filepath)

            if intervention_count is not None:
                total_interventions += intervention_count
                processed_files_count += 1
            else:
                skipped_files_count += 1

    if processed_files_count > 0:
        average_interventions = total_interventions / processed_files_count
        print(f"Processed {processed_files_count} trajectory files.")
        print(f"Skipped {skipped_files_count} failed trajectory files.")
        print(f"Total intervention segments found: {total_interventions}")
        print(f"Average number of intervention segments per processed file: {average_interventions:.2f}")
    else:
        print("No valid trajectory files were processed.")

# --- Main Execution ---

if __name__ == "__main__":
    # Define directories. Adjust these paths as needed.
    # Assuming the script is run from within the 'run_main_exp' directory or similar structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir) # Go up one level from script dir
    # Or define base_dir explicitly if needed:
    # base_dir = "/home/anthony/vlm-interventions/haco/run_main_exp"

    trajectory_dir = os.path.join(base_dir, "run_main_exp", "trajectory_data")
    # If your structure is different, adjust trajectory_dir accordingly.
    # Example: trajectory_dir = "/path/to/your/trajectory_data"

    if not os.path.isdir(trajectory_dir):
        print(f"Error: Trajectory directory not found at {trajectory_dir}")
        print("Please ensure the 'trajectory_dir' variable points to the correct location.")
    else:
        print(f"Processing trajectories in: {trajectory_dir}")
        process_directory(trajectory_dir) 