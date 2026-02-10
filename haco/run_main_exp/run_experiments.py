#!/usr/bin/env python3
"""
Run all VLM intervention experiments.

This script runs all the experiments defined in the study:

Baselines:
1. pretrained_only - Original pretrained policy (no additional training)
2. intervention_only - BC on original intervention data only
3. hg_dagger - HG-DAgger-style training (mix policy + interventions)

VLM-Corrected (using Anthropic Claude with extended thinking):
4. vlm_corrected_V_SA - VLM corrections using Vision + State/Action only
5. vlm_corrected_T_V_SA - VLM corrections using Transcript + Vision + State/Action
6. vlm_corrected_T_V_SA_pre - Same as above but including pre-intervention data

VLM Strategy Variations:
7. vlm_weighting - VLM assigns weights to data points
8. vlm_filtering - VLM filters which data points to include
9. vlm_rejection - VLM rejects certain action directions

Usage:
    # Run all experiments
    python run_experiments.py --all

    # Run specific experiments
    python run_experiments.py --experiments intervention_only vlm_corrected_T_V_SA

    # List available experiments
    python run_experiments.py --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    training_mode: str
    use_images: bool = True
    use_transcript: bool = True
    correction_strategy: str = "action_correction"
    include_pre_intervention: bool = False
    skip_correction: bool = False
    use_existing_corrected: bool = False


# Define all experiments using factorial design:
# - 3 baselines (no VLM)
# - 4 strategies × 2 input modalities = 8 VLM conditions
# - 1 extra condition (pre-intervention data)

EXPERIMENTS = {
    # ==========================================================================
    # BASELINES (no VLM processing)
    # ==========================================================================
    "pretrained_only": ExperimentConfig(
        name="pretrained_only",
        description="Original pretrained policy (no additional training)",
        training_mode="pretrained_only",
        skip_correction=True,
    ),
    "intervention_only": ExperimentConfig(
        name="intervention_only",
        description="BC on original intervention data only",
        training_mode="intervention_only",
        skip_correction=True,
    ),
    "hg_dagger": ExperimentConfig(
        name="hg_dagger",
        description="HG-DAgger-style training (mix policy + interventions)",
        training_mode="hg_dagger",
        skip_correction=True,
    ),

    # ==========================================================================
    # FACTORIAL DESIGN: Strategy × Input Modality
    # Input modalities:
    #   - V_SA: Vision + State/Action (no transcript)
    #   - T_V_SA: Transcript + Vision + State/Action
    # Strategies:
    #   - action_correction: VLM suggests better actions
    #   - weighting: VLM assigns importance weights
    #   - filtering: VLM decides keep/discard
    #   - rejection: VLM zeroes out bad action directions
    # ==========================================================================

    # --- Action Correction ---
    "correction_V_SA": ExperimentConfig(
        name="correction_V_SA",
        description="Action correction with Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=False,
        correction_strategy="action_correction",
    ),
    "correction_T_V_SA": ExperimentConfig(
        name="correction_T_V_SA",
        description="Action correction with Transcript + Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=True,
        correction_strategy="action_correction",
    ),

    # --- Weighting ---
    "weighting_V_SA": ExperimentConfig(
        name="weighting_V_SA",
        description="VLM weighting with Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=False,
        correction_strategy="weighting",
    ),
    "weighting_T_V_SA": ExperimentConfig(
        name="weighting_T_V_SA",
        description="VLM weighting with Transcript + Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=True,
        correction_strategy="weighting",
    ),

    # --- Filtering ---
    "filtering_V_SA": ExperimentConfig(
        name="filtering_V_SA",
        description="VLM filtering with Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=False,
        correction_strategy="filtering",
    ),
    "filtering_T_V_SA": ExperimentConfig(
        name="filtering_T_V_SA",
        description="VLM filtering with Transcript + Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=True,
        correction_strategy="filtering",
    ),

    # --- Rejection ---
    "rejection_V_SA": ExperimentConfig(
        name="rejection_V_SA",
        description="VLM rejection with Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=False,
        correction_strategy="rejection",
    ),
    "rejection_T_V_SA": ExperimentConfig(
        name="rejection_T_V_SA",
        description="VLM rejection with Transcript + Vision + State/Action",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=True,
        correction_strategy="rejection",
    ),

    # ==========================================================================
    # ADDITIONAL CONDITION: Pre-intervention data
    # ==========================================================================
    "correction_T_V_SA_pre": ExperimentConfig(
        name="correction_T_V_SA_pre",
        description="Action correction + pre-intervention data (T+V+SA)",
        training_mode="vlm_corrected",
        use_images=True,
        use_transcript=True,
        correction_strategy="action_correction",
        include_pre_intervention=True,
    ),
}


def build_command(exp: ExperimentConfig, num_iters: int = 1000, vlm_provider: str = "openai") -> list[str]:
    """Build the command to run an experiment."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "correct_and_train.py"),
        "--experiment-name", exp.name,
        "--training-mode", exp.training_mode,
        "--correction-strategy", exp.correction_strategy,
        "--num-iters", str(num_iters),
        "--vlm-provider", vlm_provider,
    ]

    if exp.use_images:
        cmd.append("--use-images")
    else:
        cmd.append("--no-images")

    if exp.use_transcript:
        cmd.append("--use-transcript")
    else:
        cmd.append("--no-transcript")

    if exp.include_pre_intervention:
        cmd.append("--include-pre-intervention")

    if exp.skip_correction:
        cmd.append("--skip-correction")

    if exp.use_existing_corrected:
        cmd.append("--use-existing-corrected")

    return cmd


def run_experiment(exp: ExperimentConfig, num_iters: int = 1000, dry_run: bool = False, vlm_provider: str = "openai") -> bool:
    """Run a single experiment."""
    cmd = build_command(exp, num_iters, vlm_provider)
    print(f"\n{'='*60}")
    print(f"Experiment: {exp.name}")
    print(f"Description: {exp.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment {exp.name} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: Experiment {exp.name} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM intervention experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        help="Specific experiments to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run only baseline experiments (no VLM)",
    )
    parser.add_argument(
        "--vlm-only",
        action="store_true",
        help="Run only VLM experiments",
    )
    parser.add_argument(
        "--vlm-provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="VLM provider to use (default: openai)",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:\n")
        print(f"{'Name':<30} {'Description'}")
        print("-" * 70)
        for name, exp in EXPERIMENTS.items():
            print(f"{name:<30} {exp.description}")
        return

    # Determine which experiments to run
    experiments_to_run = []

    if args.all:
        experiments_to_run = list(EXPERIMENTS.values())
    elif args.baselines_only:
        experiments_to_run = [
            EXPERIMENTS["pretrained_only"],
            EXPERIMENTS["intervention_only"],
            EXPERIMENTS["hg_dagger"],
        ]
    elif args.vlm_only:
        # All experiments that use VLM (not baselines)
        baseline_names = {"pretrained_only", "intervention_only", "hg_dagger"}
        experiments_to_run = [
            exp for name, exp in EXPERIMENTS.items()
            if name not in baseline_names
        ]
    elif args.experiments:
        experiments_to_run = [EXPERIMENTS[name] for name in args.experiments]
    else:
        parser.print_help()
        print("\nPlease specify --all, --experiments, --baselines-only, or --vlm-only")
        return

    # Run experiments
    print(f"\nRunning {len(experiments_to_run)} experiments...")
    print(f"VLM Provider: {args.vlm_provider}")
    results = {}

    for exp in experiments_to_run:
        success = run_experiment(exp, args.num_iters, args.dry_run, args.vlm_provider)
        results[exp.name] = success

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{name:<30} {status}")

    # Return non-zero if any experiment failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
