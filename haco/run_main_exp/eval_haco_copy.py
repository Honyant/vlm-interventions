import os.path
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from haco.algo.haco.haco import HACOTrainer
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train_utils import initialize_ray
from panda3d.core import PNMImage, Filename
import io
from concurrent.futures import ThreadPoolExecutor
import threading
import tensorflow as tf
import torch
from torch import nn
from bc_trainer import ImprovedBCPolicy

import gym

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def save_image(frame_path, frame):
    frame.write(Filename.from_os_specific(frame_path))
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

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
    
    # Initialize ImprovedBCPolicy
    policy = ImprovedBCPolicy(dummy_env, dummy_action_space, config)
    
    # Load state dictionary with normalization stats
    try:
        state_dict = torch.load(model_path)
        policy.network.load_state_dict(state_dict['network_state'])
        policy.obs_mean = state_dict['obs_mean']
        policy.obs_std = state_dict['obs_std']
        print("Successfully loaded model and normalization parameters")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    policy.network.eval()
    
    def _f(obs):
        try:
            with torch.no_grad():
                # Convert to numpy if tensor
                if torch.is_tensor(obs):
                    obs = obs.numpy()
                
                # Normalize observations using saved statistics
                obs_normalized = (obs - policy.obs_mean) / policy.obs_std
                
                # Compute action
                obs_tensor = torch.FloatTensor(obs_normalized)
                action = policy.network(obs_tensor)
                
                # Convert to numpy and clip to valid range
                action_np = action.numpy()
                action_np = np.clip(action_np, -1.0, 1.0)
                
                return {"default_policy": action_np}
        except Exception as e:
            print(f"Error computing actions with PyTorch model: {e}")
            print(f"Observation shape: {obs.shape}, type: {type(obs)}")
            print(f"Observation stats - min: {obs.min():.3f}, max: {obs.max():.3f}")
            if hasattr(policy, 'obs_mean'):
                print(f"Normalization stats - mean shape: {policy.obs_mean.shape}, std shape: {policy.obs_std.shape}")
            raise
            
    return _f

def get_tensorflow_function(exp_path, ckpt_idx):
    """Original TensorFlow checkpoint loading function"""
    ckpt = os.path.join(exp_path, "checkpoint_{}".format(ckpt_idx), "checkpoint-{}".format(ckpt_idx))
    
    config = {
        "env": HumanInTheLoopEnv,
        "framework": "tf",
        "model": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "env_config": {
            "manual_control": False,
            "use_render": True,
            "controller": "joystick",
            "window_size": (1600, 1100),
            "cos_similarity": True,
            "map": "CTO",
        }
    }
    
    try:
        trainer = HACOTrainer(config)
        trainer.restore(ckpt)
    except ValueError as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure the checkpoint was trained with the same model architecture")
        raise

    def _f(obs):
        try:
            ret = trainer.compute_actions({"default_policy": obs})
            return ret
        except Exception as e:
            print(f"Error computing actions: {e}")
            raise

    return _f

if __name__ == '__main__':
    # hyperparameters
    USE_PYTORCH = True  # Set to True for PyTorch model evaluation
    if USE_PYTORCH:
        MODEL_PATH = "/home/anthony/HACO/haco/run_main_exp/checkpoints/best/policy.pt"
    else:
        CKPT_PATH = "/home/anthony/HACO/haco/run_main_exp/SAC_HumanInTheLoopEnv_cff06_00000_0_seed=0_2024-05-17_03-45-28"
        CKPT_START = 1982
        CKPT_END = 1983
        
    EPISODE_NUM_PER_CKPT = 20
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

    video_dir = os.path.join("/home/anthony/HACO/haco/run_main_exp", "videos")
    os.makedirs(video_dir, exist_ok=True)

    traj_dir = os.path.join("/home/anthony/HACO/haco/run_main_exp", "trajectory_data")
    os.makedirs(traj_dir, exist_ok=True)

    if USE_PYTORCH:
        ckpt_indices = [0]  # Just use one index for PyTorch model
        compute_actions = get_pytorch_function(MODEL_PATH)
    else:
        ckpt_indices = range(CKPT_START, CKPT_END)
        compute_actions = None

    for ckpt_idx in ckpt_indices:
        if not USE_PYTORCH:
            try:
                compute_actions = get_tensorflow_function(CKPT_PATH, ckpt_idx)
            except Exception as e:
                print(f"Error loading checkpoint {ckpt_idx}: {e}")
                continue

        for episode in range(EPISODE_NUM_PER_CKPT):
            o = env.reset()
            done = False
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
                "trajectory": [],
                "frames": {}
            }

            step = 0
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

                # Handle frame capture
                frame = None
                if hasattr(env, 'engine'):
                    img = PNMImage()
                    env.engine.win.getScreenshot(img)
                    frame = img

                frame_filename = f"frame_{ckpt_idx}_{episode}_{step}.jpg"
                frame_path = os.path.join(video_dir, frame_filename)
                if frame is not None:
                    episode_data["frames"][frame_path] = frame
                    
                episode_data["trajectory"].append((o.tolist(), action_to_send.tolist(), info["takeover"], frame_path if frame is not None else None))

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

                # Save trajectory data
                traj_data = {
                    "trajectory": episode_data["trajectory"],
                    "metrics": episode_metrics
                }

                filename = f"trajectory_ckpt_{ckpt_idx}_episode_{episode}.json"
                filepath = os.path.join(traj_dir, filename)

                with open(filepath, 'w') as f:
                    json.dump(traj_data, f, cls=NumpyEncoder)

                # Save frames using thread pool
                with ThreadPoolExecutor(max_workers=threading.active_count() * 2) as executor:
                    futures = [
                        executor.submit(save_image, frame_path, frame) 
                        for frame_path, frame in episode_data["frames"].items()
                    ]
                    for future in futures:
                        future.result()

                print(f"Saved trajectory data for checkpoint {ckpt_idx}, episode {episode}")
                print(f"Saved {len(episode_data['frames'])} video frames")

                super_data[ckpt_idx].append(episode_metrics)

        if super_data[ckpt_idx]:
            print(
                f"CKPT:{ckpt_idx} | "
                f"success_rate:{np.mean([ep['success_rate'] for ep in super_data[ckpt_idx]]):.4f}, "
                f"mean_episode_reward:{np.mean([ep['raw_episode_reward'] for ep in super_data[ckpt_idx]]):.4f}, "
                f"mean_episode_cost:{np.mean([ep['cost'] for ep in super_data[ckpt_idx]]):.4f}"
            )

    env.close()

    try:
        with open("eval_haco_ret.json", "w") as f:
            json.dump(super_data, f, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving final results: {e}")