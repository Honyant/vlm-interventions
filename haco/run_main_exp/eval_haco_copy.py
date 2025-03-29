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
import argparse
import gym
import pyaudio
import wave
from datetime import datetime

class AudioRecorder:
    def __init__(self, output_dir, save_audio):
        self.output_dir = output_dir
        self.save_audio = save_audio
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None

        if self.save_audio:
            os.makedirs(output_dir, exist_ok=True)

    def callback(self, in_data, frame_count, time_info, status):
        if self.recording and self.save_audio:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        if not self.save_audio:
            return
        self.recording = True
        self.frames = []
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1,
                                      rate=44100, input=True,
                                      stream_callback=self.callback)
        self.stream.start_stream()

    def stop_recording(self, trajectory_id):
        if not self.save_audio or not self.recording:
            return None
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()

        filename = f"trajectory_{trajectory_id}.wav"
        filepath = os.path.join(self.output_dir, filename)

        wf = wave.open(filepath, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        return filepath

    def close(self):
        self.audio.terminate()

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
    dummy_env = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(259,), dtype=np.float32)
    dummy_action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    config = {
        "framework": "torch",
        "hidden_dim": 256,
        "num_hidden_layers": 2,
        "dropout_rate": 0,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
    }

    policy = ImprovedBCPolicy(dummy_env, dummy_action_space, config)

    try:
        state_dict = torch.load(model_path)
        policy.network.load_state_dict(state_dict['network_state'])
        policy.obs_mean = state_dict['obs_mean']
        policy.obs_std = state_dict['obs_std']
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    policy.network.eval()

    def _f(obs):
        try:
            with torch.no_grad():
                if torch.is_tensor(obs):
                    obs = obs.numpy()
                obs_normalized = (obs - policy.obs_mean) / policy.obs_std
                obs_tensor = torch.FloatTensor(obs_normalized)
                action = policy.network(obs_tensor)
                action_np = action.numpy()
                action_np = np.clip(action_np, -1.0, 1.0)
                return {"default_policy": action_np}
        except Exception as e:
            print(f"Error computing actions: {e}")
            raise

    return _f

def get_tensorflow_function(exp_path, ckpt_idx):
    ckpt = os.path.join(exp_path, f"checkpoint_{ckpt_idx}", f"checkpoint-{ckpt_idx}")

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

    trainer = HACOTrainer(config)
    trainer.restore(ckpt)

    def _f(obs):
        try:
            ret = trainer.compute_actions({"default_policy": obs})
            return ret
        except Exception as e:
            print(f"Error computing actions: {e}")
            raise

    return _f

def save_image(frame_path, frame):
    frame.write(Filename.from_os_specific(frame_path))

def main(args):
    USE_PYTORCH = args.use_pytorch
    SAVE_DATA = args.save_data

    if USE_PYTORCH:
        MODEL_PATH = args.pytorch_model_path
    else:
        CKPT_PATH = args.tensorflow_ckpt_path
        CKPT_START = args.ckpt_start
        CKPT_END = args.ckpt_end

    EPISODE_NUM = args.num_episodes
    RENDER = args.render

    env_config = {
        "manual_control": True,
        "use_render": True,
        "controller": "joystick",
        "window_size": (1600, 1100),
        "cos_similarity": True,
        "map": "CTO",
    }

    initialize_ray(test_mode=False, local_mode=False, num_gpus=0)

    def make_env(env_cfg=None):
        env_cfg = env_cfg or {}
        env_cfg.update(dict(manual_control=True, use_render=RENDER))
        return HumanInTheLoopEnv(env_cfg)

    super_data = defaultdict(list)
    env = make_env(env_config)

    video_dir = os.path.join(args.output_dir, "videos")
    traj_dir = os.path.join(args.output_dir, "trajectory_data")
    audio_dir = os.path.join(args.output_dir, "audio_recordings")

    if SAVE_DATA:
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

    audio_recorder = AudioRecorder(audio_dir, save_audio=SAVE_DATA)

    if USE_PYTORCH:
        ckpt_indices = [0]
        compute_actions = get_pytorch_function(MODEL_PATH)
    else:
        ckpt_indices = range(CKPT_START, CKPT_END)
        compute_actions = None

    try:
        for ckpt_idx in ckpt_indices:
            if not USE_PYTORCH:
                try:
                    compute_actions = get_tensorflow_function(CKPT_PATH, ckpt_idx)
                except Exception as e:
                    print(f"Error loading checkpoint {ckpt_idx}: {e}")
                    continue

            for episode in range(EPISODE_NUM):
                trajectory_id = f"{ckpt_idx}_{episode}"
                audio_recorder.start_recording()

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
                }

                step = 0
                trajectory = []
                frames = {}

                while not done:
                    try:
                        action_to_send = compute_actions(o)["default_policy"]
                        new_o, r, done, info = env.step(action_to_send)
                    except Exception as e:
                        print(f"Error during episode step {step}: {e}")
                        break

                    for key in ["velocity", "steering", "step_reward", "acceleration"]:
                        episode_data[key].append(info[key])

                    episode_data["takeover"] += 1 if info["takeover"] else 0
                    episode_data["raw_episode_reward"] += info["step_reward"]
                    episode_data["episode_crash_rate"] += 1 if info["crash"] else 0
                    episode_data["episode_out_of_road_rate"] += 1 if info["out_of_road"] else 0
                    episode_data["total_takeover_cost"] += info["takeover_cost"]
                    episode_data["total_native_cost"] += info["native_cost"]
                    episode_data["cost"] += info["cost"] if "cost" in info else info["native_cost"]
                    episode_data["episode_crash_vehicle"] += 1 if info["crash_vehicle"] else 0
                    episode_data["episode_crash_object"] += 1 if info["crash_object"] else 0

                    frame_filename = f"frame_{trajectory_id}_{step}.jpg"
                    frame_path = os.path.join(video_dir, frame_filename)
                    if SAVE_DATA and hasattr(env, 'engine'):
                        img = PNMImage()
                        env.engine.win.getScreenshot(img)
                        frames[frame_path] = img

                    if SAVE_DATA:
                        trajectory.append((o.tolist(), action_to_send.tolist(), info["takeover"], frame_path))

                    o = new_o
                    step += 1

                    if args.max_steps and step >= args.max_steps:
                        done = True

                audio_filepath = audio_recorder.stop_recording(trajectory_id)

                if step > 0 and SAVE_DATA:
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

                    traj_data = {
                        "trajectory": trajectory,
                        "metrics": episode_metrics,
                        "audio_file": audio_filepath
                    }

                    filepath = os.path.join(traj_dir, f"trajectory_{trajectory_id}.json")
                    with open(filepath, 'w') as f:
                        json.dump(traj_data, f, cls=NumpyEncoder)

                    with ThreadPoolExecutor(max_workers=threading.active_count() * 2) as executor:
                        futures = [
                            executor.submit(save_image, path, frame)
                            for path, frame in frames.items()
                        ]
                        for future in futures:
                            future.result()

                    super_data[ckpt_idx].append(episode_metrics)

                print(f"Episode {episode}/{EPISODE_NUM} completed")

            if super_data[ckpt_idx] and SAVE_DATA:
                print(
                    f"CKPT:{ckpt_idx} | "
                    f"success_rate:{np.mean([ep['success_rate'] for ep in super_data[ckpt_idx]]):.4f}, "
                    f"mean_episode_reward:{np.mean([ep['raw_episode_reward'] for ep in super_data[ckpt_idx]]):.4f}, "
                    f"mean_episode_cost:{np.mean([ep['cost'] for ep in super_data[ckpt_idx]]):.4f}"
                )
    finally:
        env.close()
        audio_recorder.close()

    if SAVE_DATA:
        try:
            results_file = os.path.join(args.output_dir, "eval_haco_ret.json")
            with open(results_file, "w") as f:
                json.dump(super_data, f, cls=NumpyEncoder)
            print(f"Results saved to {results_file}")
        except Exception as e:
            print(f"Error saving final results: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data collection script for VLM Interventions')
    parser.add_argument('-n', '--num-episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('-m', '--max-steps', type=int, default=None, help='Maximum steps per episode (optional)')
    parser.add_argument('--use-pytorch', action='store_true', help='Use PyTorch model instead of TensorFlow')
    parser.add_argument('--pytorch-model-path', type=str, default="/home/anthony/vlm-interventions/haco/run_main_exp/checkpoints/best/policy.pt")
    parser.add_argument('--tensorflow-ckpt-path', type=str, default="/home/anthony/vlm-interventions/haco/run_main_exp/SAC_HumanInTheLoopEnv_cff06_00000_0_seed=0_2024-05-17_03-45-28")
    parser.add_argument('--ckpt-start', type=int, default=1982)
    parser.add_argument('--ckpt-end', type=int, default=1983)
    parser.add_argument('--render', action='store_true', default=True, help='Enable rendering')
    parser.add_argument('--output-dir', type=str, default="/home/anthony/vlm-interventions/haco/run_main_exp")
    parser.add_argument('--window-size', type=int, nargs=2, default=[1600, 1100])
    parser.add_argument('--map', type=str, default="CTO")
    parser.add_argument('--save-data', action='store_true', help='Toggle saving of all outputs: trajectory, images, audio')

    args = parser.parse_args()

    if args.use_pytorch:
        if not os.path.exists(args.pytorch_model_path):
            raise ValueError(f"PyTorch model path does not exist: {args.pytorch_model_path}")
    else:
        if not os.path.exists(args.tensorflow_ckpt_path):
            raise ValueError(f"TensorFlow checkpoint path does not exist: {args.tensorflow_ckpt_path}")
        if args.ckpt_start >= args.ckpt_end:
            raise ValueError("ckpt-start must be less than ckpt-end")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
