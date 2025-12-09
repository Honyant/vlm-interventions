# Plan: Unified Correction and Training Pipeline

## Goal
Create a clean, well-structured script that:
1. Uses `episode_index.py` to discover all non-flawed episodes
2. For each episode with interventions, creates a VLM-corrected version (if not already cached)
3. Trains a BC policy on the corrected trajectories

## File: `correct_and_train.py`

---

## Module Structure

```
correct_and_train.py
├── Configuration (dataclass)
├── Correction Logic (refactored from analyze2.py)
├── Pipeline Orchestration
└── CLI Entry Point
```

---

## 1. Configuration

Create a `@dataclass` for all settings:

```python
@dataclass
class PipelineConfig:
    # Paths
    base_dir: Path
    metadata_path: Path
    corrected_dir: Path
    checkpoint_dir: Path

    # VLM correction settings
    openai_api_key: str
    context_steps: int = 10          # steps before intervention to include
    timesteps_per_second: int = 10
    max_retries: int = 5
    use_images: bool = True          # include image grid in VLM prompt
    use_transcript: bool = True      # include audio transcript in VLM prompt

    # Parallel processing
    num_workers: int = 4             # number of parallel correction workers

    # BC training settings
    num_iters: int = 1000
    batch_size: int = 256
    learning_rate: float = 1e-2
    hidden_dim: int = 256
    num_layers: int = 2
    # Loss weights
    w_delta: float = 1.0
    w_jump: float = 1.0
    w_sat: float = 0.1
    w_jac: float = 0.0
    delta_max: float = 0.15
    sat_margin: float = 0.9
    checkpoint_freq: int = 100

    # Behavior flags
    skip_correction: bool = False    # skip VLM, use existing corrected files only
    skip_training: bool = False      # only do correction, no BC training
    force_recorrect: bool = False    # re-run VLM even if corrected file exists
    use_existing_corrected: bool = False  # train on pre-existing corrected_trajectories dir
```

---

## 2. Correction Module

Refactor `analyze2.py` into clean, testable functions:

### 2.1 `InterventionSegment` dataclass
```python
@dataclass
class InterventionSegment:
    start_idx: int
    end_idx: int      # exclusive
    context_start: int  # includes pre-intervention context
```

### 2.2 `find_intervention_segments(trajectory: list) -> list[InterventionSegment]`
- Same logic as analyze2, but returns structured objects
- Filter: only segments with length > 2

### 2.3 `build_vlm_prompt(trajectory, segment, transcript_text) -> str`
- Extract and format trajectory data for the context window
- Cleaner prompt construction

### 2.4 `create_image_grid(image_paths, output_path) -> Path`
- Same as analyze2, but with better error handling
- Returns the output path on success

### 2.5 `build_vlm_request(trajectory, segment, audio_dir, video_dir, config) -> dict`
- Builds the full request payload conditionally:
  - If `config.use_images`: create image grid and encode as base64
  - If `config.use_transcript`: load and include .srt transcript text
- Returns dict with prompt text and optional image data

### 2.6 `query_vlm_for_corrections(request: dict, config) -> list[dict]`
- Makes the OpenAI API call
- Parses JSON response
- Validates correction format
- Raises clear exceptions on failure

### 2.7 `apply_corrections(trajectory, corrections) -> list`
- Apply validated corrections to trajectory
- Returns modified trajectory (non-mutating)

### 2.8 `correct_trajectory_file(src_path, dst_path, audio_dir, video_dir, config) -> bool`
- Orchestrates correction for a single file
- Returns True if corrections were made, False if no interventions found
- Handles caching: skips if dst_path exists and not force_recorrect

---

## 3. Pipeline Orchestration

### 3.1 `get_episodes_to_process(config) -> list[EpisodeAssets]`
- Uses `episode_index.build_episode_index()`
- Filters to non-flawed, non-practice episodes
- Returns list of EpisodeAssets

### 3.2 `get_corrected_path(episode: EpisodeAssets, config) -> Path`
- Computes the expected corrected trajectory path for an episode
- Format: `corrected_dir / timestamp / trajectory_0_{episode_id}.json`

### 3.3 `correct_single_episode(episode: EpisodeAssets, config) -> Path | None`
- Worker function for parallel processing
- Calls `correct_trajectory_file` with appropriate paths
- Returns corrected path on success, None on failure

### 3.4 `ensure_corrected_trajectories(episodes, config) -> list[Path]`
- Uses `concurrent.futures.ProcessPoolExecutor` with `config.num_workers`
- For each episode in parallel:
  - Check if corrected version exists in `corrected_dir`
  - If not (or force_recorrect), run correction
  - Collect all corrected trajectory paths
- Shows progress with tqdm
- Returns list of paths for BC training

### 3.5 `collect_existing_corrected(config) -> list[Path]`
- Scans `corrected_dir` for all existing corrected trajectory files
- Used when `use_existing_corrected=True`
- Returns list of all .json trajectory paths found

### 3.6 `run_pipeline(config) -> None`
Main entry point:
```python
def run_pipeline(config: PipelineConfig) -> None:
    # Mode 1: Use existing corrected trajectories directly
    if config.use_existing_corrected:
        corrected_paths = collect_existing_corrected(config)
        logger.info(f"Using {len(corrected_paths)} existing corrected trajectories")
    else:
        # Step 1: Discover episodes
        episodes = get_episodes_to_process(config)
        logger.info(f"Found {len(episodes)} non-flawed episodes")

        # Step 2: Ensure corrections exist (parallel processing)
        if not config.skip_correction:
            corrected_paths = ensure_corrected_trajectories(episodes, config)
            logger.info(f"Corrected {len(corrected_paths)} trajectories")
        else:
            corrected_paths = [get_corrected_path(ep, config) for ep in episodes
                               if get_corrected_path(ep, config).exists()]

    # Step 3: Train BC
    if not config.skip_training:
        if not corrected_paths:
            raise ValueError("No corrected trajectories found for training")
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
```

---

## 4. CLI Entry Point

Use `argparse` with sensible defaults:

```python
def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Correct intervention trajectories with VLM and train BC policy"
    )

    # Path arguments
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent / "study_data")
    parser.add_argument("--corrected-dir", type=Path, default=Path(__file__).parent / "corrected_trajectories")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).parent / "checkpoints")

    # Mode flags
    parser.add_argument("--skip-correction", action="store_true",
                        help="Skip VLM correction, use existing corrected files")
    parser.add_argument("--skip-training", action="store_true",
                        help="Only run correction, skip BC training")
    parser.add_argument("--force-recorrect", action="store_true",
                        help="Re-run VLM correction even if corrected file exists")
    parser.add_argument("--use-existing-corrected", action="store_true",
                        help="Train directly on existing corrected_trajectories dir")

    # VLM input flags
    parser.add_argument("--use-images", action="store_true", default=True,
                        help="Include image grid in VLM prompt (default: True)")
    parser.add_argument("--no-images", action="store_false", dest="use_images",
                        help="Exclude images from VLM prompt")
    parser.add_argument("--use-transcript", action="store_true", default=True,
                        help="Include audio transcript in VLM prompt (default: True)")
    parser.add_argument("--no-transcript", action="store_false", dest="use_transcript",
                        help="Exclude transcript from VLM prompt")

    # VLM settings
    parser.add_argument("--context-steps", type=int, default=10,
                        help="Number of steps before intervention to include as context")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Max retry attempts for VLM API calls")

    # Parallel processing
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers for correction")

    # BC training settings (same as bc_trainer.py)
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--w-delta", type=float, default=1.0)
    parser.add_argument("--w-jump", type=float, default=1.0)
    parser.add_argument("--w-sat", type=float, default=0.1)
    parser.add_argument("--w-jac", type=float, default=0.0)
    parser.add_argument("--delta-max", type=float, default=0.15)
    parser.add_argument("--sat-margin", type=float, default=0.9)
    parser.add_argument("--checkpoint-freq", type=int, default=100)

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
        max_retries=args.max_retries,
        use_images=args.use_images,
        use_transcript=args.use_transcript,
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
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    config = parse_args()
    run_pipeline(config)
```

---

## 5. Code Quality Guidelines

### 5.1 Type Hints
- All functions have full type annotations
- Use `from __future__ import annotations` for forward references

### 5.2 Error Handling
- Custom exceptions: `CorrectionError`, `VLMError`
- Clear error messages with context (file path, segment info)
- Graceful degradation: if one file fails, continue with others

### 5.3 Logging
- Use Python `logging` module instead of print
- Log levels: INFO for progress, WARNING for skipped files, ERROR for failures
- Include file names and counts in log messages

### 5.4 Progress Tracking
- Use `tqdm` for progress bars during correction and training
- Print summary at end: files processed, files skipped, errors

### 5.5 Documentation
- Module docstring explaining the pipeline
- Function docstrings with Args/Returns
- Inline comments for non-obvious logic (e.g., segment filtering)

---

## 6. File Organization

The corrected trajectories will be organized as:
```
corrected_trajectories/
├── {timestamp}/
│   ├── trajectory_0_0.json
│   ├── trajectory_0_1.json
│   └── ...
└── ...
```

This mirrors the source structure and makes cache checking simple.

---

## 7. Testing Considerations

Design for testability:
- Pure functions where possible (no side effects)
- Dependency injection for API calls (can mock in tests)
- Small focused functions that do one thing

---

## 8. Implementation Order

1. Create the file with imports and PipelineConfig dataclass
2. Implement `find_intervention_segments` (port from analyze2)
3. Implement `build_vlm_prompt` and `create_image_grid` (port from analyze2)
4. Implement `query_vlm_for_corrections` with proper error handling
5. Implement `apply_corrections` with validation
6. Implement `correct_trajectory_file` orchestrator
7. Implement `get_episodes_to_process` (use episode_index)
8. Implement `ensure_corrected_trajectories` with caching
9. Implement `run_pipeline` main function
10. Implement CLI with argparse
11. Add logging throughout
12. Add progress bars with tqdm

---

## 9. Example Usage

```bash
# Full pipeline: correct all episodes and train
python correct_and_train.py

# Train on existing corrected trajectories (no VLM calls)
python correct_and_train.py --use-existing-corrected

# Only run correction, skip training
python correct_and_train.py --skip-training

# Correction without images (text-only VLM prompt)
python correct_and_train.py --no-images

# Correction without transcript
python correct_and_train.py --no-transcript

# Force re-correction of all files
python correct_and_train.py --force-recorrect

# Use 8 parallel workers for faster correction
python correct_and_train.py --num-workers 8

# Custom training parameters
python correct_and_train.py --use-existing-corrected --num-iters 5000 --batch-size 512
```

---

## 10. Remaining Decisions

1. **Failed trajectories**: Should we skip trajectories where `success_rate == 0`?
   - Recommendation: Yes, add `--include-failed` flag to optionally include them

2. **Image grid sampling**: Currently samples every 3rd image. Is this the right granularity?
   - Could make configurable with `--image-sample-rate`
