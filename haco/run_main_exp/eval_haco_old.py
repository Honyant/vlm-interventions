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

def get_function(exp_path, ckpt_idx):
    ckpt = os.path.join(exp_path, "241119_checkpoint_{}".format(ckpt_idx), "checkpoint-{}".format(ckpt_idx))
    trainer = HACOTrainer(dict(env=HumanInTheLoopEnv))
    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == '__main__':
    # hyperparameters
    CKPT_PATH = "/home/anthony/HACO/haco/run_main_exp/HACO_240713-150340/HACO_HumanInTheLoopEnv_005cd_00000_0_seed=0_2024-07-13_15-03-41"
    EPISODE_NUM_PER_CKPT = 20
    CKPT_START = 53
    CKPT_END = 54
    RENDER = True
    env_config = {
        "manual_control": True,
        "use_render": True,
        "controller": "joystick",
        "window_size": (1600, 1100),
        "cos_similarity": True,
        "map": "CTO",
        "environment_num": 1,
        "start_seed": 15,
    }

    initialize_ray(test_mode=False, local_mode=False, num_gpus=0)

    def make_env(env_cfg=None):
        env_cfg = env_cfg or {}
        env_cfg.update(dict(manual_control=True, use_render=RENDER))
        return HumanInTheLoopEnv(env_cfg)

    super_data = defaultdict(list)
    env = make_env(env_config)

    video_dir = os.path.join("/home/anthony/HACO/haco/run_main_exp", "videos")
    os.makedirs(video_dir, exist_ok=True)

    traj_dir = os.path.join("/home/anthony/HACO/haco/run_main_exp", "trajectory_data")
    os.makedirs(traj_dir, exist_ok=True)

    for ckpt_idx in range(CKPT_START, CKPT_END):
        ckpt = ckpt_idx
        compute_actions = get_function(CKPT_PATH, ckpt_idx)

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
                action_to_send = compute_actions(o)["default_policy"]
                new_o, r, done, info = env.step(action_to_send)

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

                frame = None
                if hasattr(env, 'engine'):
                    img = PNMImage()
                    env.engine.win.getScreenshot(img)
                    frame = img

                frame_filename = f"frame_{ckpt}_{episode}_{step}.jpg"
                frame_path = os.path.join(video_dir, frame_filename)
                episode_data["frames"][frame_path] = frame
                    
                episode_data["trajectory"].append((o.tolist(), action_to_send.tolist(), info["takeover"], frame_path if frame is not None else None))

                o = new_o
                step += 1

            # Calculate episode metrics
            episode_length = step
            arrive_dest = info["arrive_dest"]
            crash = info["crash"]
            out_of_road = info["out_of_road"]
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
                "overtake_num": int(info["overtake_vehicle_num"]),
                "episode_crash_vehicle_num": float(episode_data["episode_crash_vehicle"]),
                "episode_crash_object_num": float(episode_data["episode_crash_object"]),
            }

            # Save trajectory data
            traj_data = {
                "trajectory": episode_data["trajectory"],
                "metrics": episode_metrics
            }

            filename = f"trajectory_ckpt_{ckpt}_episode_{episode}.json"
            filepath = os.path.join(traj_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(traj_data, f, cls=NumpyEncoder)

            # Save buffered images using a thread pool
            with ThreadPoolExecutor(max_workers=threading.active_count() * 2) as executor:
                futures = [executor.submit(save_image, frame_path, frame) 
                        for frame_path, frame in episode_data["frames"].items()]
                for future in futures:
                    future.result()  # Wait for all tasks to complete

            print(f"Saved {len(episode_data['frames'])} video frames for checkpoint {ckpt}, episode {episode} in {video_dir}")

            print(f"Saved {len(episode_data['frames'])} video frames for checkpoint {ckpt}, episode {episode} in {video_dir}")

            print(f"Saved trajectory data for checkpoint {ckpt}, episode {episode} to {filepath}")
            print(f"Saved {len(episode_data['frames'])} video frames for checkpoint {ckpt}, episode {episode} in {video_dir}")

            super_data[ckpt].append(episode_metrics)

        print(
            f"CKPT:{ckpt} | "
            f"success_rate:{np.mean([ep['success_rate'] for ep in super_data[ckpt]]):.4f}, "
            f"mean_episode_reward:{np.mean([ep['raw_episode_reward'] for ep in super_data[ckpt]]):.4f}, "
            f"mean_episode_cost:{np.mean([ep['cost'] for ep in super_data[ckpt]]):.4f}"
        )

    env.close()

    try:
        with open("eval_haco_ret.json", "w") as f:
            json.dump(super_data, f, cls=NumpyEncoder)
    except:
        pass

    print(super_data)