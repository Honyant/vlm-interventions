import os
import numpy as np
from collections import defaultdict
import json
from haco.algo.haco.haco import HACOTrainer
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train_utils import initialize_ray
import torch
from torch import nn
from bc_trainer import ImprovedBCPolicy
import gym


def get_pytorch_function(model_path):
    """Load improved PyTorch model and return compute actions function"""
    # Create dummy environment for spaces
    dummy_env = gym.spaces.Box(
        low=-np.inf, high=np.inf, 
        shape=(259,), dtype=np.float32)
    dummy_action_space = gym.spaces.Box(
        low=-1, high=1, 
        shape=(2,), dtype=np.float32)
    
    # Create policy with improved config
    config = {
        "framework": "torch",
        "hidden_dim": 256,
        "num_hidden_layers": 2,
        "dropout_rate": 0,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
    }
    
    def _f(obs):
        return {"default_policy": np.array([0.0, 0.0])}
            
    return _f

if __name__ == '__main__':
    # hyperparameters
    MODEL_PATH = "/home/anthony/HACO/haco/run_main_exp/checkpoints/best/policy.pt"
    EPISODE_NUM_PER_CKPT = 1
    RENDER = True
    
    env_config = {
        "manual_control": True,
        "use_render": True,
        "controller": "joystick",
        "window_size": (1600, 1100),
        "cos_similarity": True,
        "map": "CTO",
    }

    try:
        initialize_ray(test_mode=False, local_mode=False, num_gpus=0)
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        raise

    def make_env(env_cfg=None):
        env_cfg = env_cfg or {}
        env_cfg.update(dict(manual_control=True, use_render=RENDER))
        return HumanInTheLoopEnv(env_cfg)

    super_data = defaultdict(list)
    
    try:
        env = make_env(env_config)
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise

    compute_actions = get_pytorch_function(MODEL_PATH)

    for episode in range(EPISODE_NUM_PER_CKPT):
        o = env.reset()
        done = False
        step = 0
        
        episode_data = {
            "velocity": [],
            "steering": [],
            "step_reward": [],
            "acceleration": [],
            "takeover": 0,
            "raw_episode_reward": 0,
            "episode_crash_rate": 0,
            "episode_out_of_road_rate": 0,
            "high_speed_rate": 0,
            "total_takeover_cost": 0,
            "total_native_cost": 0,
            "cost": 0,
            "episode_crash_vehicle": 0,
            "episode_crash_object": 0,
        }
        
        while not done:
            try:
                action_to_send = compute_actions(o)["default_policy"]
                new_o, r, done, info = env.step(action_to_send)
            except Exception as e:
                print(f"Error during episode step {step}: {e}")
                break

            # Update episode data
            episode_data["velocity"].append(info["velocity"])
            episode_data["steering"].append(info["steering"])
            episode_data["step_reward"].append(info["step_reward"])
            episode_data["acceleration"].append(info["acceleration"])
            episode_data["takeover"] += 1 if info["takeover"] else 0
            episode_data["raw_episode_reward"] += info["step_reward"]
            episode_data["episode_crash_rate"] += 1 if info["crash"] else 0
            episode_data["episode_out_of_road_rate"] += 1 if info["out_of_road"] else 0
            episode_data["total_takeover_cost"] += info["takeover_cost"]
            episode_data["total_native_cost"] += info["native_cost"]
            episode_data["cost"] += info["cost"] if "cost" in info else info["native_cost"]
            episode_data["episode_crash_vehicle"] += 1 if info["crash_vehicle"] else 0
            episode_data["episode_crash_object"] += 1 if info["crash_object"] else 0

            o = new_o
            step += 1

        if step > 0:
            # Calculate episode metrics
            episode_length = step
            arrive_dest = info.get("arrive_dest", False)
            crash = info.get("crash", False)
            out_of_road = info.get("out_of_road", False)
            max_step_rate = not (arrive_dest or crash or out_of_road)

            episode_metrics = {
                "success_rate": float(arrive_dest),
                "crash_rate": float(crash),
                "out_of_road_rate": float(out_of_road),
                "max_step_rate": float(max_step_rate),
                "velocity_max": float(np.max(episode_data["velocity"])),
                "velocity_mean": float(np.mean(episode_data["velocity"])),
                "velocity_min": float(np.min(episode_data["velocity"])),
                "steering_max": float(np.max(episode_data["steering"])),
                "steering_mean": float(np.mean(episode_data["steering"])),
                "steering_min": float(np.min(episode_data["steering"])),
                "acceleration_min": float(np.min(episode_data["acceleration"])),
                "acceleration_mean": float(np.mean(episode_data["acceleration"])),
                "acceleration_max": float(np.max(episode_data["acceleration"])),
                "step_reward_max": float(np.max(episode_data["step_reward"])),
                "step_reward_mean": float(np.mean(episode_data["step_reward"])),
                "step_reward_min": float(np.min(episode_data["step_reward"])),
                "takeover_rate": float(episode_data["takeover"] / episode_length),
                "takeover_count": float(episode_data["takeover"]),
                "raw_episode_reward": float(episode_data["raw_episode_reward"]),
                "episode_crash_num": float(episode_data["episode_crash_rate"]),
                "episode_out_of_road_num": float(episode_data["episode_out_of_road_rate"]),
                "high_speed_rate": float(episode_data["high_speed_rate"] / episode_length),
                "total_takeover_cost": float(episode_data["total_takeover_cost"]),
                "total_native_cost": float(episode_data["total_native_cost"]),
                "cost": float(episode_data["cost"]),
                "overtake_num": int(info.get("overtake_vehicle_num", 0)),
                "episode_crash_vehicle_num": float(episode_data["episode_crash_vehicle"]),
                "episode_crash_object_num": float(episode_data["episode_crash_object"]),
            }

            super_data[0].append(episode_metrics)

        if super_data[0]:
            print(
                f"success_rate:{np.mean([ep['success_rate'] for ep in super_data[0]]):.4f}, "
                f"mean_episode_reward:{np.mean([ep['raw_episode_reward'] for ep in super_data[0]]):.4f}, "
                f"mean_episode_cost:{np.mean([ep['cost'] for ep in super_data[0]]):.4f}"
            )

    env.close()