# filename: print_trajectory_info.py

import os
import json
import argparse
from PIL import Image

def load_trajectory(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def print_trajectory_info(traj_data, max_items=5):
    print(f"Trajectory Information:")
    print(f"Number of steps: {len(traj_data['trajectory'])}")
    
    print("\nTrajectory (first {} and last {}):".format(max_items, max_items))
    for i in list(range(min(max_items, len(traj_data['trajectory'])))) + list(range(-min(max_items, len(traj_data['trajectory'])), 0)):
        obs, action, frame_path, takeover = traj_data['trajectory'][i]
        print(f"Step {i}:")
        print(f"  Observation: {obs}")
        print(f"  Action: {action}")
        print(f"  Frame: {frame_path}")
        print(f"  Takeover: {takeover}")
    
    print("\nTakeovers:")
    takeover_count = sum(step[3] for step in traj_data['trajectory'])
    print(f"Total takeovers: {takeover_count}")
    print(f"Takeover rate: {takeover_count / len(traj_data['trajectory']):.2%}")
    
    print("\nTakeover steps:")
    for i, (_, _, _, takeover) in enumerate(traj_data['trajectory']):
        if takeover:
            print(f"Step {i}")

def main():
    parser = argparse.ArgumentParser(description="Print trajectory information from saved JSON files.")
    parser.add_argument("log_dir", help="Path to the log directory containing the trajectory_data folder")
    parser.add_argument("--episode", type=int, help="Specific episode number to print. If not provided, prints info for all episodes.")
    parser.add_argument("--show-images", action="store_true", help="Display rendered frames")
    args = parser.parse_args()

    traj_dir = os.path.join(args.log_dir, "trajectory_data")
    
    if not os.path.exists(traj_dir):
        print(f"Error: Trajectory data directory not found at {traj_dir}")
        return

    if args.episode:
        file_path = os.path.join(traj_dir, f"trajectory_episode_{args.episode}.json")
        if os.path.exists(file_path):
            traj_data = load_trajectory(file_path)
            print(f"\nTrajectory for Episode {args.episode}:")
            print_trajectory_info(traj_data)
            if args.show_images:
                for i, (_, _, frame_path, _) in enumerate(traj_data['trajectory']):
                    Image.open(frame_path).show(title=f"Episode {args.episode}, Step {i}")
        else:
            print(f"Error: Trajectory file for episode {args.episode} not found.")
    else:
        for filename in sorted(os.listdir(traj_dir)):
            if filename.startswith("trajectory_episode_") and filename.endswith(".json"):
                file_path = os.path.join(traj_dir, filename)
                traj_data = load_trajectory(file_path)
                episode_num = filename.split("_")[-1].split(".")[0]
                print(f"\nTrajectory for Episode {episode_num}:")
                print_trajectory_info(traj_data)
                print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()