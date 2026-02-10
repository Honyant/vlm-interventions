"""Data collection script for VLM Interventions.

Runs a policy in MetaDrive, records trajectories to JSON, and prints summary stats.
Works headless (no display) by default; pass --render for a GUI window.
"""

import argparse
import json
import os
import os.path
import wave
from collections import defaultdict

import numpy as np

from study_utils import (
    NumpyEncoder,
    accumulate_step,
    compute_episode_metrics,
    load_pytorch_model,
    load_tensorflow_model,
    make_env,
    new_episode_data,
    print_summary,
)

try:
    import pyaudio
    HAS_PYAUDIO = True
except (ImportError, OSError):
    HAS_PYAUDIO = False


# ---------------------------------------------------------------------------
# Audio recorder (optional, graceful when pyaudio is unavailable)
# ---------------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, output_dir, save_audio):
        self.output_dir = output_dir
        self.save_audio = save_audio and HAS_PYAUDIO
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio() if HAS_PYAUDIO else None
        self.stream = None

        if self.save_audio:
            os.makedirs(output_dir, exist_ok=True)
        elif save_audio and not HAS_PYAUDIO:
            print("WARNING: pyaudio not available, audio recording disabled")

    def callback(self, in_data, frame_count, time_info, status):
        if self.recording and self.save_audio:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        if not self.save_audio:
            return
        self.recording = True
        self.frames = []
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=44100,
            input=True, stream_callback=self.callback,
        )
        self.stream.start_stream()

    def stop_recording(self, trajectory_id):
        if not self.save_audio or not self.recording:
            return None
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()

        filename = f"trajectory_{trajectory_id}.wav"
        filepath = os.path.join(self.output_dir, filename)

        wf = wave.open(filepath, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(self.frames))
        wf.close()
        return filepath

    def close(self):
        if self.audio:
            self.audio.terminate()


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------
def main(args):
    save_data = args.save_data

    # --- Load model ----------------------------------------------------------
    if args.use_pytorch:
        ckpt_indices = [0]
        compute_actions = load_pytorch_model(args.pytorch_model_path)
    else:
        from haco.utils.train_utils import initialize_ray
        initialize_ray(test_mode=False, local_mode=False, num_gpus=0)
        ckpt_indices = range(args.ckpt_start, args.ckpt_end)
        compute_actions = None

    # --- Create env ----------------------------------------------------------
    env = make_env(render=args.render, map_name=args.map, window_size=args.window_size)

    # --- Output directories --------------------------------------------------
    traj_dir = os.path.join(args.output_dir, "trajectory_data")
    audio_dir = os.path.join(args.output_dir, "audio_recordings")
    if save_data:
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

    audio_recorder = AudioRecorder(audio_dir, save_audio=save_data)

    super_data = defaultdict(list)

    try:
        for ckpt_idx in ckpt_indices:
            # Load TF checkpoint per-index (PyTorch already loaded above)
            if not args.use_pytorch:
                try:
                    compute_actions = load_tensorflow_model(
                        args.tensorflow_ckpt_path, ckpt_idx, use_render=args.render,
                    )
                except Exception as e:
                    print(f"Error loading checkpoint {ckpt_idx}: {e}")
                    continue

            for episode in range(args.num_episodes):
                trajectory_id = f"{ckpt_idx}_{episode}"
                audio_recorder.start_recording()

                o = env.reset()
                done = False
                episode_data = new_episode_data()
                step = 0
                trajectory = []

                while not done:
                    try:
                        action_to_send = compute_actions(o)["default_policy"]
                        new_o, r, done, info = env.step(action_to_send)
                    except Exception as e:
                        print(f"Error during episode step {step}: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                    accumulate_step(episode_data, info)

                    if save_data:
                        trajectory.append((
                            o.tolist(),
                            action_to_send.tolist(),
                            info["takeover"],
                        ))

                    o = new_o
                    step += 1

                    if args.max_steps and step >= args.max_steps:
                        done = True

                audio_filepath = audio_recorder.stop_recording(trajectory_id)

                if step > 0:
                    episode_metrics = compute_episode_metrics(episode_data, info, step)
                    super_data[ckpt_idx].append(episode_metrics)

                    if save_data:
                        traj_data = {
                            "trajectory": trajectory,
                            "metrics": episode_metrics,
                            "audio_file": audio_filepath,
                        }
                        filepath = os.path.join(traj_dir, f"trajectory_{trajectory_id}.json")
                        with open(filepath, "w") as f:
                            json.dump(traj_data, f, cls=NumpyEncoder)

                print(f"Episode {episode}/{args.num_episodes} completed")

            # Per-checkpoint summary
            if super_data[ckpt_idx]:
                print(
                    f"CKPT:{ckpt_idx} | "
                    f"success_rate:{np.mean([ep['success_rate'] for ep in super_data[ckpt_idx]]):.4f}, "
                    f"mean_episode_reward:{np.mean([ep['raw_episode_reward'] for ep in super_data[ckpt_idx]]):.4f}, "
                    f"mean_episode_cost:{np.mean([ep['cost'] for ep in super_data[ckpt_idx]]):.4f}"
                )
    finally:
        env.close()
        audio_recorder.close()

    # --- Overall summary -----------------------------------------------------
    all_episodes = [ep for episodes in super_data.values() for ep in episodes]
    print_summary(all_episodes)

    if save_data:
        results_file = os.path.join(args.output_dir, "eval_haco_ret.json")
        with open(results_file, "w") as f:
            json.dump(super_data, f, cls=NumpyEncoder)
        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection script for VLM Interventions")
    parser.add_argument("-n", "--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("-m", "--max-steps", type=int, default=None, help="Maximum steps per episode")
    parser.add_argument("--use-pytorch", action="store_true", help="Use PyTorch model instead of TensorFlow")
    parser.add_argument("--pytorch-model-path", type=str,
                        default="/home/anthony/vlm-interventions/haco/run_main_exp/checkpoints/best/policy.pt")
    parser.add_argument("--tensorflow-ckpt-path", type=str,
                        default="/home/anthony/vlm-interventions/haco/run_main_exp/study_assets/"
                                "SAC_HumanInTheLoopEnv_cff06_00000_0_seed=0_2024-05-17_03-45-28")
    parser.add_argument("--ckpt-start", type=int, default=1982)
    parser.add_argument("--ckpt-end", type=int, default=1983)
    parser.add_argument("--render", action="store_true", default=True,
                        help="Enable rendering (disabled by default for headless)")
    parser.add_argument("--output-dir", type=str,
                        default="/home/anthony/vlm-interventions/haco/run_main_exp")
    parser.add_argument("--window-size", type=int, nargs=2, default=[1600, 1100])
    parser.add_argument("--map", type=str, default="CTO")
    parser.add_argument("--save-data", action="store_true",
                        help="Save trajectory JSON and audio recordings")

    args = parser.parse_args()

    if args.use_pytorch:
        if not os.path.exists(args.pytorch_model_path):
            raise ValueError(f"PyTorch model path does not exist: {args.pytorch_model_path}")
    else:
        if not os.path.exists(args.tensorflow_ckpt_path):
            raise ValueError(f"TensorFlow checkpoint path does not exist: {args.tensorflow_ckpt_path}")
        if args.ckpt_start >= args.ckpt_end:
            raise ValueError("ckpt-start must be less than ckpt-end")

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
