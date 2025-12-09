"""
Unified pipeline for VLM-based trajectory correction and behavioral cloning training.

This script:
1. Discovers non-flawed episodes from study data using episode_index
2. For episodes with human interventions, uses GPT-4o to suggest corrected actions
3. Caches corrected trajectories to avoid redundant API calls
4. Trains a BC policy on the corrected data

Usage:
    # Full pipeline
    python correct_and_train.py

    # Train on existing corrected trajectories only
    python correct_and_train.py --use-existing-corrected

    # Correction without images
    python correct_and_train.py --no-images --skip-training
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from study_data.episode_index import EpisodeAssets, build_episode_index

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the correction and training pipeline."""

    # Paths
    base_dir: Path
    metadata_path: Path
    corrected_dir: Path
    checkpoint_dir: Path

    # VLM correction settings
    openai_api_key: str = ""
    context_steps: int = 10
    timesteps_per_second: int = 10
    max_retries: int = 5
    use_images: bool = True
    use_transcript: bool = True
    image_sample_rate: int = 3  # sample every Nth image for grid

    # Parallel processing
    num_workers: int = 4

    # BC training settings
    num_iters: int = 1000
    batch_size: int = 256
    learning_rate: float = 1e-2
    hidden_dim: int = 256
    num_layers: int = 2
    w_delta: float = 1.0
    w_jump: float = 1.0
    w_sat: float = 0.1
    w_jac: float = 0.0
    delta_max: float = 0.15
    sat_margin: float = 0.9
    checkpoint_freq: int = 100

    # Behavior flags
    skip_correction: bool = False
    skip_training: bool = False
    force_recorrect: bool = False
    use_existing_corrected: bool = False
    include_failed: bool = False  # include trajectories with success_rate=0


# =============================================================================
# Exceptions
# =============================================================================

class CorrectionError(Exception):
    """Raised when trajectory correction fails."""
    pass


class VLMError(CorrectionError):
    """Raised when VLM API call fails."""
    pass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InterventionSegment:
    """Represents a contiguous segment where intervention occurred."""
    start_idx: int
    end_idx: int  # exclusive
    context_start: int  # includes pre-intervention context

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


# =============================================================================
# Intervention Detection
# =============================================================================

def find_intervention_segments(
    trajectory: list,
    context_steps: int = 10,
    min_length: int = 3,
) -> list[InterventionSegment]:
    """
    Find contiguous segments where intervention_occurring is True.

    Args:
        trajectory: List of [observation, action, intervention_occurring, image_file]
        context_steps: Number of steps before intervention to include as context
        min_length: Minimum segment length to include

    Returns:
        List of InterventionSegment objects
    """
    segments = []
    i = 0
    n = len(trajectory)

    while i < n:
        # Check if intervention is occurring (index 2 in each record)
        if trajectory[i][2] is True:
            start = i
            while i < n and trajectory[i][2] is True:
                i += 1
            end = i

            # Only include segments longer than min_length
            if (end - start) >= min_length:
                context_start = max(0, start - context_steps)
                segments.append(InterventionSegment(
                    start_idx=start,
                    end_idx=end,
                    context_start=context_start,
                ))
        else:
            i += 1

    return segments


# =============================================================================
# Transcript Handling
# =============================================================================

def load_transcript(srt_path: Path) -> list | None:
    """Load SRT transcript file using pysrt."""
    if not srt_path.exists():
        return None
    try:
        import pysrt
        return pysrt.open(str(srt_path))
    except ImportError:
        logger.warning("pysrt not installed, skipping transcript loading")
        return None
    except Exception as e:
        logger.warning(f"Failed to load transcript {srt_path}: {e}")
        return None


def get_transcript_for_range(
    subs: list,
    start_time: float,
    end_time: float,
) -> str:
    """
    Extract transcript text for a time range.

    Args:
        subs: List of pysrt subtitle entries
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Concatenated transcript text for the range
    """
    if subs is None:
        return ""

    lines = []
    for sub in subs:
        sub_start = (
            sub.start.hours * 3600
            + sub.start.minutes * 60
            + sub.start.seconds
            + sub.start.milliseconds / 1000.0
        )
        sub_end = (
            sub.end.hours * 3600
            + sub.end.minutes * 60
            + sub.end.seconds
            + sub.end.milliseconds / 1000.0
        )
        if sub_end >= start_time and sub_start <= end_time:
            lines.append(sub.text)

    return "\n".join(lines)


# =============================================================================
# Image Handling
# =============================================================================

def create_image_grid(
    image_paths: list[Path],
    output_path: Path,
    num_cols: int = 3,
) -> Path:
    """
    Create a grid image from a list of image paths.

    Args:
        image_paths: List of paths to images
        output_path: Where to save the grid image
        num_cols: Number of columns in the grid

    Returns:
        Path to the created grid image
    """
    if not image_paths:
        raise ValueError("No images provided for grid creation")

    images = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(img.size for img in images))

    max_width = max(widths)
    max_height = max(heights)
    num_rows = (len(images) + num_cols - 1) // num_cols

    grid = Image.new("RGB", (max_width * num_cols, max_height * num_rows))

    for idx, image in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        grid.paste(image, (col * max_width, row * max_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def encode_image_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =============================================================================
# VLM Prompt Building
# =============================================================================

def build_trajectory_context(
    trajectory: list,
    segment: InterventionSegment,
) -> str:
    """Build text representation of trajectory for the context window."""
    lines = []
    for idx in range(segment.context_start, segment.end_idx):
        action = trajectory[idx][1]
        intervention = trajectory[idx][2]
        lines.append(f"Index {idx}: action={action}, intervention={intervention}")
    return "\n".join(lines)


def build_vlm_prompt(
    trajectory: list,
    segment: InterventionSegment,
    transcript_text: str = "",
    use_transcript: bool = True,
) -> str:
    """
    Build the prompt for the VLM to analyze and correct the trajectory.

    Args:
        trajectory: Full trajectory data
        segment: The intervention segment to analyze
        transcript_text: Optional transcript text for the time range
        use_transcript: Whether to include transcript in prompt

    Returns:
        Formatted prompt string
    """
    trajectory_text = build_trajectory_context(trajectory, segment)

    prompt = f"""Trajectory data for context window (indices {segment.context_start} to {segment.end_idx}):
{trajectory_text}
"""

    if use_transcript and transcript_text:
        prompt += f"""
Transcript of human speech during this segment:
{transcript_text}
"""

    prompt += f"""
Analyze the sequence of driving actions before and during the intervention. For each action frame in this context window:
1. Provide a brief text description suggesting better pre-intervention actions.
2. Output a JSON array with objects containing the index and corrected (steering, acceleration) pairs for that index.
- Use negative acceleration values for deceleration.
- After values must be in [-1,1]
- Include 3 significant figures.
- Start from index {segment.context_start}.
- Cover all time steps in this context window.
- If the intervention is not happening, just return the original action.
Example format YOU MUST FOLLOW (the vectors for steering mean before and after (after is what you change it to)):
```json
[
  {{
    "index": 192,
    "steering": [0.071, 0.150],
    "acceleration": [0.999, 0.500]
  }},
  ...
]
```"""
    return prompt


def build_vlm_request(
    trajectory: list,
    segment: InterventionSegment,
    audio_dir: Path | None,
    video_dir: Path | None,
    config: PipelineConfig,
    trajectory_name: str,
) -> dict[str, Any]:
    """
    Build the complete VLM request with optional images and transcript.

    Args:
        trajectory: Full trajectory data
        segment: The intervention segment
        audio_dir: Directory containing audio/transcript files
        video_dir: Directory containing frame images
        config: Pipeline configuration
        trajectory_name: Base name of the trajectory file (for temp file naming)

    Returns:
        Dict with 'prompt' and optionally 'image_base64' keys
    """
    # Get transcript if enabled
    transcript_text = ""
    if config.use_transcript and audio_dir:
        srt_path = audio_dir / f"{trajectory_name}.srt"
        subs = load_transcript(srt_path)
        if subs:
            start_time = segment.context_start / config.timesteps_per_second
            end_time = segment.end_idx / config.timesteps_per_second
            transcript_text = get_transcript_for_range(subs, start_time, end_time)

    prompt = build_vlm_prompt(
        trajectory,
        segment,
        transcript_text,
        use_transcript=config.use_transcript,
    )

    result: dict[str, Any] = {"prompt": prompt}

    # Create image grid if enabled
    if config.use_images and video_dir:
        image_paths = []
        for idx in range(segment.context_start, segment.end_idx):
            if idx < len(trajectory) and len(trajectory[idx]) > 3:
                image_file = trajectory[idx][3]
                if image_file and Path(image_file).exists():
                    image_paths.append(Path(image_file))

        if image_paths:
            # Sample images at the configured rate
            sampled_images = image_paths[:: config.image_sample_rate]
            if sampled_images:
                grid_path = (
                    config.corrected_dir
                    / "temp_grids"
                    / f"grid_{trajectory_name}_{segment.start_idx}_{segment.end_idx}.jpg"
                )
                try:
                    create_image_grid(sampled_images, grid_path)
                    result["image_base64"] = encode_image_base64(grid_path)
                except Exception as e:
                    logger.warning(f"Failed to create image grid: {e}")

    return result


# =============================================================================
# VLM API Interaction
# =============================================================================

def query_vlm_for_corrections(
    request: dict[str, Any],
    config: PipelineConfig,
) -> list[dict]:
    """
    Query GPT-4o for trajectory corrections.

    Args:
        request: Dict with 'prompt' and optionally 'image_base64'
        config: Pipeline configuration

    Returns:
        List of correction dicts with 'index', 'steering', 'acceleration'

    Raises:
        VLMError: If API call fails after all retries
    """
    content: list[dict] = [{"type": "text", "text": request["prompt"]}]

    if "image_base64" in request:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{request['image_base64']}"},
        })

    messages = [{"role": "user", "content": content}]

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 10000,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.openai_api_key}",
    }

    last_error = None
    for attempt in range(config.max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            response_json = response.json()
            response_content = response_json["choices"][0]["message"]["content"]

            # Extract JSON array from response
            json_start = response_content.find("[")
            json_end = response_content.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                raise VLMError("No JSON array found in VLM response")

            json_str = response_content[json_start:json_end]
            corrections = json.loads(json_str)

            # Validate corrections format
            validate_corrections(corrections)
            return corrections

        except requests.RequestException as e:
            last_error = VLMError(f"API request failed: {e}")
            logger.warning(f"Attempt {attempt + 1}/{config.max_retries} failed: {e}")
        except json.JSONDecodeError as e:
            last_error = VLMError(f"Failed to parse JSON response: {e}")
            logger.warning(f"Attempt {attempt + 1}/{config.max_retries} failed: {e}")
        except Exception as e:
            last_error = VLMError(f"Unexpected error: {e}")
            logger.warning(f"Attempt {attempt + 1}/{config.max_retries} failed: {e}")

    raise last_error or VLMError("All retry attempts failed")


def validate_corrections(corrections: list[dict]) -> None:
    """
    Validate the format of VLM corrections.

    Raises:
        VLMError: If corrections format is invalid
    """
    for corr in corrections:
        if "index" not in corr:
            raise VLMError(f"Missing 'index' in correction: {corr}")
        if "steering" not in corr or "acceleration" not in corr:
            raise VLMError(f"Missing steering/acceleration in correction: {corr}")

        steering = corr["steering"]
        acceleration = corr["acceleration"]

        if not (isinstance(steering, list) and len(steering) == 2):
            raise VLMError(f"Invalid steering format: {steering}")
        if not (isinstance(acceleration, list) and len(acceleration) == 2):
            raise VLMError(f"Invalid acceleration format: {acceleration}")

        # Validate values are in range
        for val in [steering[1], acceleration[1]]:
            if not isinstance(val, (int, float)) or not (-1 <= val <= 1):
                raise VLMError(f"Value out of range [-1, 1]: {val}")


# =============================================================================
# Correction Application
# =============================================================================

def apply_corrections(
    trajectory: list,
    corrections: list[dict],
    segment: InterventionSegment,
) -> list:
    """
    Apply VLM corrections to trajectory data.

    Args:
        trajectory: Original trajectory data (will be deep copied)
        corrections: List of correction dicts from VLM
        segment: The intervention segment being corrected

    Returns:
        New trajectory with corrections applied
    """
    import copy
    new_trajectory = copy.deepcopy(trajectory)

    for corr in corrections:
        idx = corr["index"]
        if idx < segment.context_start or idx >= segment.end_idx:
            continue

        new_steering = corr["steering"][1]
        new_acceleration = corr["acceleration"][1]

        # Update action: [steering, acceleration]
        new_trajectory[idx][1] = [new_steering, new_acceleration]

    return new_trajectory


# =============================================================================
# Single File Correction
# =============================================================================

def correct_trajectory_file(
    src_path: Path,
    dst_path: Path,
    audio_dir: Path | None,
    video_dir: Path | None,
    config: PipelineConfig,
) -> bool:
    """
    Correct a single trajectory file using VLM.

    Args:
        src_path: Source trajectory file
        dst_path: Destination for corrected trajectory
        audio_dir: Directory with audio/transcript files
        video_dir: Directory with frame images
        config: Pipeline configuration

    Returns:
        True if corrections were made, False if no interventions found
    """
    # Check cache
    if dst_path.exists() and not config.force_recorrect:
        logger.debug(f"Using cached: {dst_path.name}")
        return True

    # Load trajectory
    with open(src_path) as f:
        data = json.load(f)

    trajectory = data.get("trajectory", [])
    if not trajectory:
        logger.warning(f"No trajectory data in {src_path.name}")
        return False

    # Check for failed trajectory
    if not config.include_failed:
        success_rate = data.get("metrics", {}).get("success_rate", 1)
        if success_rate == 0:
            logger.debug(f"Skipping failed trajectory: {src_path.name}")
            return False

    # Find intervention segments
    segments = find_intervention_segments(trajectory, config.context_steps)

    if not segments:
        # No interventions - copy file as-is
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w") as f:
            json.dump(data, f, indent=2)
        return True

    # Process each segment
    trajectory_name = src_path.stem
    corrected_trajectory = trajectory

    for segment in segments:
        try:
            request = build_vlm_request(
                corrected_trajectory,
                segment,
                audio_dir,
                video_dir,
                config,
                trajectory_name,
            )
            corrections = query_vlm_for_corrections(request, config)
            corrected_trajectory = apply_corrections(
                corrected_trajectory, corrections, segment
            )
            logger.info(
                f"Corrected segment {segment.start_idx}-{segment.end_idx} in {src_path.name}"
            )
        except CorrectionError as e:
            logger.error(f"Failed to correct {src_path.name} segment {segment.start_idx}-{segment.end_idx}: {e}")
            raise

    # Save corrected trajectory
    data["trajectory"] = corrected_trajectory
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(data, f, indent=2)

    return True


# =============================================================================
# Episode Processing
# =============================================================================

def get_episodes_to_process(config: PipelineConfig) -> list[EpisodeAssets]:
    """
    Get list of non-flawed, non-practice episodes to process.

    Args:
        config: Pipeline configuration

    Returns:
        List of EpisodeAssets to process
    """
    index = build_episode_index(config.base_dir, config.metadata_path)

    episodes = []
    for bundle in index.values():
        for episode in bundle.episodes:
            # Skip practice and flawed episodes
            if episode.practice or episode.flawed:
                continue
            episodes.append(episode)

    return episodes


def get_corrected_path(episode: EpisodeAssets, config: PipelineConfig) -> Path:
    """Get the expected path for a corrected trajectory."""
    # Extract timestamp from trajectory path
    # e.g., trajectory_data_092920251621/trajectory_0_0.json
    timestamp = episode.trajectory.parent.name.replace("trajectory_data_", "")
    return config.corrected_dir / timestamp / episode.trajectory.name


def get_audio_dir_for_episode(episode: EpisodeAssets, config: PipelineConfig) -> Path | None:
    """Get the audio directory for an episode."""
    timestamp = episode.trajectory.parent.name.replace("trajectory_data_", "")
    audio_dir = config.base_dir / f"audio_recordings_{timestamp}"
    return audio_dir if audio_dir.exists() else None


def get_video_dir_for_episode(episode: EpisodeAssets, config: PipelineConfig) -> Path | None:
    """Get the video directory for an episode."""
    timestamp = episode.trajectory.parent.name.replace("trajectory_data_", "")
    video_dir = config.base_dir / f"videos_{timestamp}"
    if video_dir.exists():
        return video_dir
    # Handle typo in folder name
    video_dir = config.base_dir / f"vieos_{timestamp}"
    return video_dir if video_dir.exists() else None


def correct_single_episode(
    episode: EpisodeAssets,
    config: PipelineConfig,
) -> Path | None:
    """
    Worker function to correct a single episode.

    Args:
        episode: Episode to correct
        config: Pipeline configuration

    Returns:
        Path to corrected file on success, None on failure
    """
    dst_path = get_corrected_path(episode, config)
    audio_dir = get_audio_dir_for_episode(episode, config)
    video_dir = get_video_dir_for_episode(episode, config)

    try:
        success = correct_trajectory_file(
            episode.trajectory,
            dst_path,
            audio_dir,
            video_dir,
            config,
        )
        return dst_path if success else None
    except Exception as e:
        logger.error(f"Failed to correct {episode.trajectory.name}: {e}")
        return None


def ensure_corrected_trajectories(
    episodes: list[EpisodeAssets],
    config: PipelineConfig,
) -> list[Path]:
    """
    Ensure all episodes have corrected trajectories, using parallel processing.

    Args:
        episodes: List of episodes to process
        config: Pipeline configuration

    Returns:
        List of paths to corrected trajectories
    """
    corrected_paths = []

    if config.num_workers <= 1:
        # Sequential processing
        for episode in tqdm(episodes, desc="Correcting trajectories"):
            result = correct_single_episode(episode, config)
            if result:
                corrected_paths.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(correct_single_episode, ep, config): ep
                for ep in episodes
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Correcting trajectories",
            ):
                result = future.result()
                if result:
                    corrected_paths.append(result)

    return corrected_paths


def collect_existing_corrected(config: PipelineConfig) -> list[Path]:
    """
    Collect all existing corrected trajectory files.

    Args:
        config: Pipeline configuration

    Returns:
        List of paths to existing corrected trajectories
    """
    if not config.corrected_dir.exists():
        return []

    paths = []
    for json_file in config.corrected_dir.rglob("trajectory_*.json"):
        paths.append(json_file)

    return sorted(paths)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(config: PipelineConfig) -> None:
    """
    Run the complete correction and training pipeline.

    Args:
        config: Pipeline configuration
    """
    # Mode 1: Use existing corrected trajectories directly
    if config.use_existing_corrected:
        corrected_paths = collect_existing_corrected(config)
        logger.info(f"Using {len(corrected_paths)} existing corrected trajectories")
    else:
        # Step 1: Discover episodes
        episodes = get_episodes_to_process(config)
        logger.info(f"Found {len(episodes)} non-flawed, non-practice episodes")

        if not episodes:
            logger.warning("No episodes to process")
            return

        # Step 2: Ensure corrections exist
        if not config.skip_correction:
            corrected_paths = ensure_corrected_trajectories(episodes, config)
            logger.info(f"Processed {len(corrected_paths)} trajectories")
        else:
            corrected_paths = [
                get_corrected_path(ep, config)
                for ep in episodes
                if get_corrected_path(ep, config).exists()
            ]
            logger.info(f"Found {len(corrected_paths)} existing corrected trajectories")

    # Step 3: Train BC
    if not config.skip_training:
        if not corrected_paths:
            raise ValueError("No corrected trajectories found for training")

        logger.info(f"Starting BC training on {len(corrected_paths)} trajectories")

        from bc_trainer import run_bc_training

        run_bc_training(
            data_source=[str(p) for p in corrected_paths],
            num_iters=config.num_iters,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            w_delta=config.w_delta,
            w_jump=config.w_jump,
            w_sat=config.w_sat,
            delta_max=config.delta_max,
            sat_margin=config.sat_margin,
            w_jac=config.w_jac,
            checkpoint_freq=config.checkpoint_freq,
        )

        logger.info("Training complete")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> PipelineConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Correct intervention trajectories with VLM and train BC policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Path arguments
    default_base = _SCRIPT_DIR / "study_data"
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help="Root directory containing study data",
    )
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=_SCRIPT_DIR / "corrected_trajectories",
        help="Directory for corrected trajectories",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_SCRIPT_DIR / "checkpoints",
        help="Directory for model checkpoints",
    )

    # Mode flags
    parser.add_argument(
        "--skip-correction",
        action="store_true",
        help="Skip VLM correction, use existing corrected files",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only run correction, skip BC training",
    )
    parser.add_argument(
        "--force-recorrect",
        action="store_true",
        help="Re-run VLM correction even if corrected file exists",
    )
    parser.add_argument(
        "--use-existing-corrected",
        action="store_true",
        help="Train directly on existing corrected_trajectories dir",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include trajectories with success_rate=0",
    )

    # VLM input flags
    parser.add_argument(
        "--use-images",
        action="store_true",
        default=True,
        help="Include image grid in VLM prompt",
    )
    parser.add_argument(
        "--no-images",
        action="store_false",
        dest="use_images",
        help="Exclude images from VLM prompt",
    )
    parser.add_argument(
        "--use-transcript",
        action="store_true",
        default=True,
        help="Include audio transcript in VLM prompt",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_false",
        dest="use_transcript",
        help="Exclude transcript from VLM prompt",
    )
    parser.add_argument(
        "--image-sample-rate",
        type=int,
        default=3,
        help="Sample every Nth image for the grid",
    )

    # VLM settings
    parser.add_argument(
        "--context-steps",
        type=int,
        default=10,
        help="Number of steps before intervention to include as context",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retry attempts for VLM API calls",
    )

    # Parallel processing
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for correction",
    )

    # BC training settings
    parser.add_argument("--num-iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--w-delta", type=float, default=1.0, help="Delta-matching loss weight")
    parser.add_argument("--w-jump", type=float, default=1.0, help="Jump-hinge loss weight")
    parser.add_argument("--w-sat", type=float, default=0.1, help="Saturation loss weight")
    parser.add_argument("--w-jac", type=float, default=0.0, help="Jacobian penalty weight")
    parser.add_argument("--delta-max", type=float, default=0.15, help="Max action delta")
    parser.add_argument("--sat-margin", type=float, default=0.9, help="Saturation margin")
    parser.add_argument("--checkpoint-freq", type=int, default=100, help="Checkpoint frequency")

    args = parser.parse_args()

    # Get API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key and not args.skip_correction and not args.use_existing_corrected:
        parser.error("OPENAI_API_KEY environment variable required for VLM correction")

    return PipelineConfig(
        base_dir=args.base_dir,
        metadata_path=args.base_dir / "traj_metadata.json",
        corrected_dir=args.corrected_dir,
        checkpoint_dir=args.checkpoint_dir,
        openai_api_key=openai_api_key,
        context_steps=args.context_steps,
        timesteps_per_second=10,
        max_retries=args.max_retries,
        use_images=args.use_images,
        use_transcript=args.use_transcript,
        image_sample_rate=args.image_sample_rate,
        num_workers=args.num_workers,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        w_delta=args.w_delta,
        w_jump=args.w_jump,
        w_sat=args.w_sat,
        w_jac=args.w_jac,
        delta_max=args.delta_max,
        sat_margin=args.sat_margin,
        checkpoint_freq=args.checkpoint_freq,
        skip_correction=args.skip_correction,
        skip_training=args.skip_training,
        force_recorrect=args.force_recorrect,
        use_existing_corrected=args.use_existing_corrected,
        include_failed=args.include_failed,
    )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
