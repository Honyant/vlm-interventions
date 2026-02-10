"""Offline evaluation script: load saved trajectory JSONs and compute summary stats.

No environment or model needed -- pure offline analysis.

Usage:
    python evaluate.py --data-dir /path/to/trajectory_data
    python evaluate.py --data-dir /path/to/trajectory_data --save-summary summary.json
"""

import argparse
import glob
import json
import os

import numpy as np


def load_metrics_from_dir(data_dir):
    """Load metrics dicts from all trajectory JSON files in *data_dir*."""
    pattern = os.path.join(data_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .json files found in {data_dir}")

    all_metrics = []
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
        if "metrics" in data:
            all_metrics.append(data["metrics"])
        else:
            print(f"WARNING: {os.path.basename(fpath)} has no 'metrics' key, skipping")
    return all_metrics, files


def print_summary(episodes):
    """Print a formatted evaluation summary."""
    if not episodes:
        print("No episodes to summarise.")
        return

    def _mean(key):
        vals = [e[key] for e in episodes if key in e]
        return np.mean(vals) if vals else float("nan")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total episodes:      {len(episodes)}")
    print(f"Success rate:        {_mean('success_rate'):.4f}")
    print(f"Crash rate:          {_mean('crash_rate'):.4f}")
    print(f"Out of road rate:    {_mean('out_of_road_rate'):.4f}")
    print(f"Max-step rate:       {_mean('max_step_rate'):.4f}")
    print(f"Mean episode reward: {_mean('raw_episode_reward'):.4f}")
    print(f"Mean episode cost:   {_mean('cost'):.4f}")
    print(f"Mean takeover rate:  {_mean('takeover_rate'):.4f}")
    print(f"Mean takeover count: {_mean('takeover_count'):.4f}")
    print(f"Mean velocity:       {_mean('velocity_mean'):.4f}")
    print(f"Mean steering:       {_mean('steering_mean'):.4f}")
    print(f"Mean acceleration:   {_mean('acceleration_mean'):.4f}")

    if any("episode_length" in e for e in episodes):
        print(f"Mean episode length: {_mean('episode_length'):.1f}")

    print("=" * 60)


def build_aggregate(episodes):
    """Return an aggregate dict suitable for saving."""
    def _mean(key):
        vals = [e[key] for e in episodes if key in e]
        return float(np.mean(vals)) if vals else None

    def _std(key):
        vals = [e[key] for e in episodes if key in e]
        return float(np.std(vals)) if vals else None

    keys = [
        "success_rate", "crash_rate", "out_of_road_rate", "max_step_rate",
        "raw_episode_reward", "cost", "takeover_rate", "takeover_count",
        "velocity_mean", "steering_mean", "acceleration_mean",
        "total_takeover_cost", "total_native_cost",
        "episode_crash_vehicle_num", "episode_crash_object_num",
        "episode_length",
    ]

    agg = {"num_episodes": len(episodes)}
    for k in keys:
        agg[f"{k}_mean"] = _mean(k)
        agg[f"{k}_std"] = _std(k)

    return agg


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation of saved trajectory data")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing trajectory_*.json files")
    parser.add_argument("--save-summary", type=str, default=None,
                        help="Optional path to save aggregate summary JSON")
    args = parser.parse_args()

    episodes, files = load_metrics_from_dir(args.data_dir)
    print(f"Loaded {len(episodes)} episodes from {len(files)} files in {args.data_dir}")

    print_summary(episodes)

    if args.save_summary:
        agg = build_aggregate(episodes)
        with open(args.save_summary, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nSummary saved to {args.save_summary}")


if __name__ == "__main__":
    main()
