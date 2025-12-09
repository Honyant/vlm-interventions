from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class EpisodeAssets:
    episode_id: int
    participant: str
    practice: bool
    flawed: bool
    trajectory: Path
    audio: Dict[str, Path]
    transcript: Optional[Path]
    frames: List[Path]

    def to_dict(self) -> Dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "participant": self.participant,
            "practice": self.practice,
            "flawed": self.flawed,
            "trajectory": str(self.trajectory),
            "audio": {fmt: str(path) for fmt, path in self.audio.items()},
            "transcript": str(self.transcript) if self.transcript else None,
            "frames": [str(frame) for frame in self.frames],
        }

@dataclass
class TimestampBundle:
    timestamp: str
    participant: str
    practice_default: bool
    flawed_default: bool
    bad_eps: List[int]
    episodes: List[EpisodeAssets] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "participant": self.participant,
            "practice": self.practice_default,
            "flawed": self.flawed_default,
            "bad_eps": self.bad_eps,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

def build_episode_index(base_dir: Path, metadata_path: Path) -> Dict[str, TimestampBundle]:
    metadata = json.loads(metadata_path.read_text())
    index: Dict[str, TimestampBundle] = {}
    for timestamp, meta in metadata.items():
        bundle = TimestampBundle(
            timestamp=timestamp,
            participant=meta["name"],
            practice_default=meta["practice"],
            flawed_default=meta["flawed"],
            bad_eps=[int(ep) for ep in meta.get("bad_eps", [])],
        )
        data_dir = base_dir / f"trajectory_data_{timestamp}"
        if not data_dir.exists():
            continue
        audio_dir = _resolve_optional_dir(base_dir, f"audio_recordings_{timestamp}")
        video_dir = _resolve_optional_dir(base_dir, f"videos_{timestamp}") or _resolve_optional_dir(
            base_dir, f"vieos_{timestamp}"
        )
        for traj_path in sorted(data_dir.glob("trajectory_0_*.json")):
            episode_id = int(traj_path.stem.split("_")[-1])
            audio, transcript = _collect_audio_paths(audio_dir, episode_id)
            frames = _collect_frame_paths(video_dir, episode_id)
            bundle.episodes.append(
                EpisodeAssets(
                    episode_id=episode_id,
                    participant=bundle.participant,
                    practice=bundle.practice_default,
                    flawed=bundle.flawed_default or episode_id in bundle.bad_eps,
                    trajectory=traj_path,
                    audio=audio,
                    transcript=transcript,
                    frames=frames,
                )
            )
        index[timestamp] = bundle
    return index

def index_to_dict(index: Dict[str, TimestampBundle]) -> Dict[str, object]:
    return {timestamp: bundle.to_dict() for timestamp, bundle in index.items()}

def _resolve_optional_dir(base_dir: Path, name: str) -> Optional[Path]:
    candidate = base_dir / name
    return candidate if candidate.exists() else None

def _collect_audio_paths(audio_dir: Optional[Path], episode_id: int) -> (Dict[str, Path], Optional[Path]):
    audio: Dict[str, Path] = {}
    transcript: Optional[Path] = None
    if not audio_dir:
        return audio, transcript
    base = f"trajectory_0_{episode_id}"
    for ext in ("mp3", "wav"):
        path = audio_dir / f"{base}.{ext}"
        if path.exists():
            audio[ext] = path
    srt_path = audio_dir / f"{base}.srt"
    if srt_path.exists():
        transcript = srt_path
    return audio, transcript

def _collect_frame_paths(video_dir: Optional[Path], episode_id: int) -> List[Path]:
    if not video_dir:
        return []
    frames, seen = [], set()
    for pattern in (
        f"frame_{episode_id}_*.jpg",
        f"frame_{episode_id}_*.png",
        f"frame_*_{episode_id}_*.jpg",
        f"frame_*_{episode_id}_*.png",
    ):
        for path in sorted(video_dir.glob(pattern)):
            if path not in seen:
                frames.append(path)
                seen.add(path)
    return frames

def _non_flawed_trajectories(index: Dict[str, TimestampBundle]) -> List[Path]:
    return [
        episode.trajectory
        for bundle in index.values()
        for episode in bundle.episodes
        if not episode.flawed
    ]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an index of study episodes and optionally train BC on non-flawed data."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Root directory containing trajectory_data_*, audio_recordings_*, and videos_* folders.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).parent / "traj_metadata.json",
        help="Path to traj_metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the resulting JSON index. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--train-non-flawed",
        action="store_true",
        help="Train the behavioral cloning policy on all non-flawed episodes.",
    )
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--w-delta", type=float, default=1.0)
    parser.add_argument("--w-jump", type=float, default=1.0)
    parser.add_argument("--w-sat", type=float, default=0.1)
    parser.add_argument("--delta-max", type=float, default=0.15)
    parser.add_argument("--sat-margin", type=float, default=0.9)
    parser.add_argument("--w-jac", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    index = build_episode_index(args.base_dir, args.metadata)
    if args.train_non_flawed:
        non_flawed = _non_flawed_trajectories(index)
        if not non_flawed:
            raise ValueError("No non-flawed episodes found.")
        parent_dir = Path(__file__).resolve().parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from bc_trainer import run_bc_training
        run_bc_training(
            data_source=[str(path) for path in non_flawed],
            num_iters=args.num_iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            w_delta=args.w_delta,
            w_jump=args.w_jump,
            w_sat=args.w_sat,
            delta_max=args.delta_max,
            sat_margin=args.sat_margin,
            w_jac=args.w_jac,
            checkpoint_freq=args.checkpoint_freq,
        )
        return
    payload = index_to_dict(index)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
