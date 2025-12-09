import json
import statistics
import argparse
import os

def analyze_results(file_path):
    """
    Analyzes the episode results from a JSON file.

    Args:
        file_path (str): The path to the JSON results file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Assuming the relevant data is the value of the first key
    if not data:
        print("Error: JSON data is empty.")
        return

    first_key = next(iter(data))
    episodes = data[first_key]

    if not isinstance(episodes, list) or not episodes:
        print(f"Error: Expected a non-empty list under key '{first_key}'. Found: {type(episodes)}")
        return

    # Dynamically get all keys from the first episode as potential metrics
    if isinstance(episodes[0], dict):
        all_keys = list(episodes[0].keys())
    else:
        print("Error: First episode data is not a dictionary.")
        return

    # Filter for keys that likely represent numerical metrics for analysis
    # Exclude keys that might not be suitable for mean/stdev calculation if needed
    metrics_to_analyze = [
        key for key in all_keys
        if isinstance(episodes[0].get(key), (int, float))
    ]

    print(f"Analyzing {len(episodes)} episodes from {file_path}")
    print(f"Metrics found: {', '.join(metrics_to_analyze)}")
    print("-" * 30)

    results = {}
    for metric in metrics_to_analyze:
        values = [episode.get(metric) for episode in episodes if episode.get(metric) is not None]
        if values:
            mean_val = statistics.mean(values)
            # Calculate stdev only if more than one data point
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0
            results[metric] = (mean_val, stdev_val)
            print(f"{metric:<20}: Mean={mean_val:>8.3f}, StdDev={stdev_val:>8.3f}")
        else:
            print(f"{metric:<20}: No data found")

    print("-" * 30)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze episode results from a JSON file.")
    parser.add_argument(
        "file_path",
        type=str,
        nargs='?',
        default="eval_haco_ret copy.json",
        help="Path to the JSON results file (default: eval_haco_ret.json)"
    )
    args = parser.parse_args()

    analyze_results(args.file_path) 