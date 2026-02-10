"""Shared utilities for data collection and evaluation scripts."""

import json
import os
from collections import defaultdict

import gym
import numpy as np
import torch

from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.run_main_exp.bc_trainer import ImprovedBCPolicy


# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_pytorch_model(model_path):
    """Load a PyTorch BC policy and return a callable: obs (np array) -> dict."""
    dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(259,), dtype=np.float32)
    dummy_act_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    config = {
        "framework": "torch",
        "hidden_dim": 256,
        "num_hidden_layers": 2,
        "dropout_rate": 0,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
    }

    policy = ImprovedBCPolicy(dummy_obs_space, dummy_act_space, config)

    state_dict = torch.load(model_path, map_location="cpu")
    policy.network.load_state_dict(state_dict["network_state"])
    policy.obs_mean = state_dict["obs_mean"]
    policy.obs_std = state_dict["obs_std"]
    policy.network.eval()

    def _f(obs):
        with torch.no_grad():
            if torch.is_tensor(obs):
                obs = obs.numpy()
            obs_normalized = (obs - policy.obs_mean) / policy.obs_std
            obs_tensor = torch.FloatTensor(obs_normalized)
            action = policy.network(obs_tensor)
            action_np = np.clip(action.numpy(), -1.0, 1.0)
            return {"default_policy": action_np}

    return _f


def load_tensorflow_model(ckpt_path, ckpt_idx, use_render=False):
    """Load a TensorFlow HACO checkpoint and return a callable: obs -> dict."""
    from haco.algo.haco.haco import HACOTrainer
    from haco.utils.train_utils import initialize_ray

    initialize_ray(test_mode=False, local_mode=False, num_gpus=0)

    ckpt = os.path.join(ckpt_path, f"checkpoint_{ckpt_idx}", f"checkpoint-{ckpt_idx}")

    config = {
        "env": HumanInTheLoopEnv,
        "framework": "tf",
        "model": {"fcnet_hiddens": [], "fcnet_activation": "relu"},
        "env_config": {
            "manual_control": False,
            "use_render": use_render,
            "controller": "keyboard",
            "window_size": (1600, 1100),
            "cos_similarity": True,
            "map": "CTO",
        },
    }

    trainer = HACOTrainer(config)
    trainer.restore(ckpt)

    def _f(obs):
        return trainer.compute_actions({"default_policy": obs})

    return _f


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(render=False, map_name="CTO", window_size=(1600, 1100)):
    """Create a HumanInTheLoopEnv with correct headless/render config.

    Key fix: offscreen_render is always False. We never need image observations;
    setting it True would trigger Panda3D's offscreen graphics pipeline (EGL/X11)
    and change obs from numpy arrays to dicts, breaking the policy.
    """
    env_config = {
        "manual_control": render,
        "use_render": render,
        "offscreen_render": False,
        "controller": "joystick" if render else "keyboard",
        "window_size": tuple(window_size),
        "cos_similarity": True,
        "map": map_name,
    }
    return HumanInTheLoopEnv(env_config)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_episode_metrics(episode_data, info, episode_length):
    """Compute summary metrics dict from accumulated episode_data, final info, and length."""
    arrive_dest = info.get("arrive_dest", False)
    crash = info.get("crash", False)
    out_of_road = info.get("out_of_road", False)
    max_step_rate = not (arrive_dest or crash or out_of_road)

    def _safe_stat(key, fn):
        vals = episode_data[key]
        return float(fn(vals)) if vals else 0.0

    return {
        "success_rate": float(arrive_dest),
        "crash_rate": float(crash),
        "out_of_road_rate": float(out_of_road),
        "max_step_rate": float(max_step_rate),
        "episode_length": episode_length,
        "velocity_max": _safe_stat("velocity", np.max),
        "velocity_mean": _safe_stat("velocity", np.mean),
        "velocity_min": _safe_stat("velocity", np.min),
        "steering_max": _safe_stat("steering", np.max),
        "steering_mean": _safe_stat("steering", np.mean),
        "steering_min": _safe_stat("steering", np.min),
        "acceleration_min": _safe_stat("acceleration", np.min),
        "acceleration_mean": _safe_stat("acceleration", np.mean),
        "acceleration_max": _safe_stat("acceleration", np.max),
        "step_reward_max": _safe_stat("step_reward", np.max),
        "step_reward_mean": _safe_stat("step_reward", np.mean),
        "step_reward_min": _safe_stat("step_reward", np.min),
        "takeover_rate": float(episode_data["takeover"] / episode_length) if episode_length else 0.0,
        "takeover_count": float(episode_data["takeover"]),
        "raw_episode_reward": float(episode_data["raw_episode_reward"]),
        "episode_crash_num": float(episode_data["episode_crash_rate"]),
        "episode_out_of_road_num": float(episode_data["episode_out_of_road_rate"]),
        "high_speed_rate": float(episode_data["high_speed_rate"] / episode_length) if episode_length else 0.0,
        "total_takeover_cost": float(episode_data["total_takeover_cost"]),
        "total_native_cost": float(episode_data["total_native_cost"]),
        "cost": float(episode_data["cost"]),
        "overtake_num": int(info.get("overtake_vehicle_num", 0)),
        "episode_crash_vehicle_num": float(episode_data["episode_crash_vehicle"]),
        "episode_crash_object_num": float(episode_data["episode_crash_object"]),
    }


def new_episode_data():
    """Return a fresh episode accumulator dict."""
    return {
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


def accumulate_step(episode_data, info):
    """Update episode_data accumulators with one step's info dict."""
    for key in ("velocity", "steering", "step_reward", "acceleration"):
        episode_data[key].append(info[key])
    episode_data["takeover"] += 1 if info["takeover"] else 0
    episode_data["raw_episode_reward"] += info["step_reward"]
    episode_data["episode_crash_rate"] += 1 if info["crash"] else 0
    episode_data["episode_out_of_road_rate"] += 1 if info["out_of_road"] else 0
    episode_data["total_takeover_cost"] += info["takeover_cost"]
    episode_data["total_native_cost"] += info["native_cost"]
    episode_data["cost"] += info.get("cost", info["native_cost"])
    episode_data["episode_crash_vehicle"] += 1 if info["crash_vehicle"] else 0
    episode_data["episode_crash_object"] += 1 if info["crash_object"] else 0


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(all_episodes):
    """Print a formatted evaluation summary table."""
    if not all_episodes:
        print("No episodes to summarise.")
        return

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Success rate:       {np.mean([e['success_rate'] for e in all_episodes]):.4f}")
    print(f"Crash rate:         {np.mean([e['crash_rate'] for e in all_episodes]):.4f}")
    print(f"Out of road rate:   {np.mean([e['out_of_road_rate'] for e in all_episodes]):.4f}")
    print(f"Mean episode reward:{np.mean([e['raw_episode_reward'] for e in all_episodes]):.4f}")
    print(f"Mean episode cost:  {np.mean([e['cost'] for e in all_episodes]):.4f}")
    print(f"Mean takeover rate: {np.mean([e['takeover_rate'] for e in all_episodes]):.4f}")
    print(f"Mean takeover count:{np.mean([e['takeover_count'] for e in all_episodes]):.4f}")
    print(f"Mean velocity:      {np.mean([e['velocity_mean'] for e in all_episodes]):.4f}")
    print("=" * 60)
