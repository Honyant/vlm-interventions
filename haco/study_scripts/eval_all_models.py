#!/usr/bin/env python3
"""
Parallel headless evaluation of all trained models.

Discovers every best/policy.pt checkpoint, runs N episodes per model in
parallel (one MetaDrive env per worker), and prints a summary table.

Usage:
    python eval_all_models.py                # defaults: 10 episodes, 4 workers
    python eval_all_models.py -n 20 -w 6     # 20 episodes, 6 parallel workers
    python eval_all_models.py --list          # just list discovered models
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_MAIN_DIR = SCRIPT_DIR.parent / "run_main_exp"
CHECKPOINT_DIR = RUN_MAIN_DIR / "checkpoints"
PRETRAINED_DIR = RUN_MAIN_DIR / "study_assets" / "checkpoints"


def discover_models():
    """Find all best/policy.pt checkpoints and return {name: path} dict."""
    models = {}

    # Pretrained original
    pt = PRETRAINED_DIR / "best" / "policy.pt"
    if pt.exists():
        models["pretrained_original"] = str(pt)

    # Everything under checkpoints/<name>/best/policy.pt
    if CHECKPOINT_DIR.exists():
        for best_dir in sorted(CHECKPOINT_DIR.glob("*/best")):
            policy_pt = best_dir / "policy.pt"
            if policy_pt.exists():
                name = best_dir.parent.name
                # The top-level checkpoints/best/ is the base BC model
                if name == "checkpoints":
                    continue
                models[name] = str(policy_pt)

    return models


def evaluate_model(name, model_path, num_episodes, map_name):
    """Run num_episodes headless and return metrics list. Runs in a subprocess."""
    # These imports are inside the function so each process gets its own copy
    # and MetaDrive engines don't collide.
    sys.path.insert(0, str(SCRIPT_DIR))
    from study_utils import (
        accumulate_step,
        compute_episode_metrics,
        load_pytorch_model,
        make_env,
        new_episode_data,
    )

    try:
        compute_actions = load_pytorch_model(model_path)
        env = make_env(render=False, map_name=map_name)
    except Exception as e:
        return name, "ERROR", str(e), []

    episodes = []
    try:
        for ep in range(num_episodes):
            o = env.reset()
            done = False
            episode_data = new_episode_data()
            step = 0
            info = {}

            while not done:
                try:
                    action = compute_actions(o)["default_policy"]
                    new_o, r, done, info = env.step(action)
                except Exception as e:
                    print(f"[{name}] step error ep {ep} step {step}: {e}")
                    break

                accumulate_step(episode_data, info)
                o = new_o
                step += 1

            if step > 0:
                metrics = compute_episode_metrics(episode_data, info, step)
                episodes.append(metrics)
    except Exception as e:
        return name, "ERROR", str(e), episodes
    finally:
        env.close()

    return name, "OK", "", episodes


def print_table(results, key_order=None):
    """Print a formatted results table."""
    if not results:
        print("No results.")
        return

    # Columns to display
    cols = [
        ("success", "success_rate"),
        ("crash", "crash_rate"),
        ("oor", "out_of_road_rate"),
        ("reward", "raw_episode_reward"),
        ("cost", "cost"),
        ("vel", "velocity_mean"),
        ("tkover", "takeover_rate"),
        ("len", "episode_length"),
    ]

    header = f"{'Model':<32}" + "".join(f"{c[0]:>10}" for c in cols)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    ordered = key_order if key_order else sorted(results.keys())
    for name in ordered:
        if name not in results:
            continue
        entry = results[name]
        if entry["status"] != "OK":
            print(f"{name:<32}  {entry['status']}: {entry.get('error', '')}")
            continue
        eps = entry["episodes"]
        vals = []
        for _, metric_key in cols:
            v = [e[metric_key] for e in eps if metric_key in e]
            vals.append(np.mean(v) if v else float("nan"))
        row = f"{name:<32}" + "".join(f"{v:>10.3f}" for v in vals)
        print(row)

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Parallel headless evaluation of all models")
    parser.add_argument("-n", "--num-episodes", type=int, default=10)
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("--map", type=str, default="CTO")
    parser.add_argument("--list", action="store_true", help="List models and exit")
    parser.add_argument("--models", nargs="+", default=None, help="Only evaluate these models")
    parser.add_argument("--save", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    models = discover_models()

    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}

    if args.list:
        print(f"\nDiscovered {len(models)} models:\n")
        for name, path in sorted(models.items()):
            print(f"  {name:<32} {path}")
        return

    if not models:
        print("No models found.")
        return

    print(f"Evaluating {len(models)} models x {args.num_episodes} episodes "
          f"({args.workers} parallel workers)\n")

    results = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_model, name, path, args.num_episodes, args.map): name
            for name, path in models.items()
        }

        for future in as_completed(futures):
            name, status, error, episodes = future.result()
            results[name] = {
                "status": status,
                "error": error,
                "episodes": episodes,
                "model_path": models[name],
            }
            n_ok = len(episodes)
            if status == "OK":
                sr = np.mean([e["success_rate"] for e in episodes]) if episodes else 0
                print(f"  [{status}] {name:<32} {n_ok} eps, success={sr:.2f}")
            else:
                print(f"  [{status}] {name:<32} {error}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s\n")
    print_table(results)

    # Save JSON
    save_path = args.save or str(RUN_MAIN_DIR / "eval_results" / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        # Strip non-serializable bits
        out = {}
        for name, entry in results.items():
            out[name] = {
                "status": entry["status"],
                "error": entry["error"],
                "model_path": entry["model_path"],
                "num_episodes": len(entry["episodes"]),
                "episodes": entry["episodes"],
            }
            if entry["episodes"]:
                agg = {}
                for k in entry["episodes"][0]:
                    vals = [e[k] for e in entry["episodes"]]
                    agg[k + "_mean"] = float(np.mean(vals))
                    agg[k + "_std"] = float(np.std(vals))
                out[name]["aggregate"] = agg
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
