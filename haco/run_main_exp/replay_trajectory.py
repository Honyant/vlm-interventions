import os
import json
import numpy as np
import argparse
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train_utils import initialize_ray

def load_trajectory(trajectory_file):
    with open(trajectory_file, 'r') as f:
        return json.load(f)

def create_replay_policy(trajectory_data):
    """Creates a policy function that replays actions from saved trajectory"""
    trajectory = trajectory_data["trajectory"]
    current_step = [0]  # Using list to allow modification in closure
    
    def policy_fn(obs):
        if current_step[0] >= len(trajectory):
            return {"default_policy": np.zeros(2)}  # Default action if we run out of trajectory
            
        # Get the saved action for this step
        _, action, _, _ = trajectory[current_step[0]]
        current_step[0] += 1
        
        return {"default_policy": np.array(action)}
        
    return policy_fn

def main(args):
    # Load trajectory data
    trajectory_data = load_trajectory(args.trajectory_file)
    
    # Environment configuration
    env_config = {
        "manual_control": False,  # Set to False since we're replaying
        "use_render": args.render,
        "controller": "keyboard",  # Doesn't matter since we're replaying
        "window_size": (1600, 1100),
        "cos_similarity": True,
        "map": "CTO",
    }

    initialize_ray(test_mode=False, local_mode=False, num_gpus=0)

    # Create environment
    env = HumanInTheLoopEnv(env_config)
    
    # Create replay policy
    compute_actions = create_replay_policy(trajectory_data)

    try:
        o = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        while not done:
            # Get action from saved trajectory
            action_to_send = compute_actions(o)["default_policy"]
            
            # Take step in environment
            o, r, done, info = env.step(action_to_send)
            episode_reward += r
            step += 1
            
            if args.max_steps and step >= args.max_steps:
                done = True
                
            # Print step information
            print(f"Step {step}: Reward = {r:.3f}, Done = {done}")
            
        print(f"\nReplay completed:")
        print(f"Total steps: {step}")
        print(f"Total reward: {episode_reward:.3f}")
        
    except Exception as e:
        print(f"An error occurred during replay: {e}")
        raise
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Replay Script')
    parser.add_argument('--trajectory-file', type=str, required=True,
                      help='Path to the trajectory JSON file to replay')
    parser.add_argument('--render', action='store_true', default=True,
                      help='Enable rendering')
    parser.add_argument('--max-steps', type=int, default=None,
                      help='Maximum steps to replay (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.trajectory_file):
        raise ValueError(f"Trajectory file does not exist: {args.trajectory_file}")
        
    main(args) 