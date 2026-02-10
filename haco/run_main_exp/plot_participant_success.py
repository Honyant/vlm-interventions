#!/usr/bin/env python3
"""
Plot participant success rates from the driving study.

Usage:
    python plot_participant_success.py
    python plot_participant_success.py --output my_graph.png
    python plot_participant_success.py --include-practice
    python plot_participant_success.py --no-anonymize  # Show real names
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def anonymize_participants(participant_data: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """
    Anonymize participant names to P01, P02, etc.

    Returns:
        Tuple of (anonymized_data, name_mapping) where name_mapping is {original: anonymous}
    """
    # Sort names alphabetically for consistent ID assignment
    sorted_names = sorted(participant_data.keys())

    # Create mapping: original name -> P01, P02, etc.
    name_mapping = {name: f"P{i+1:02d}" for i, name in enumerate(sorted_names)}

    # Create new dict with anonymized keys
    anonymized_data = {
        name_mapping[name]: episodes
        for name, episodes in participant_data.items()
    }

    return anonymized_data, name_mapping


def load_participant_data(
    base_dir: Path,
    metadata_path: Path,
    include_practice: bool = False,
    include_flawed: bool = False,
) -> dict[str, list[dict]]:
    """Load success rate data for all participants."""
    with open(metadata_path) as f:
        metadata = json.load(f)

    timestamp_info = {}
    for ts, info in metadata.items():
        timestamp_info[ts] = {
            'name': info['name'],
            'practice': info['practice'],
            'flawed': info['flawed'],
            'bad_eps': info.get('bad_eps', [])
        }

    participant_data = defaultdict(list)

    for ts, info in timestamp_info.items():
        traj_dir = base_dir / f'trajectory_data_{ts}'
        if not traj_dir.exists():
            continue

        for traj_file in sorted(traj_dir.glob('trajectory_0_*.json')):
            try:
                episode_id = int(traj_file.stem.split('_')[-1])

                with open(traj_file) as f:
                    data = json.load(f)

                metrics = data.get('metrics', {})
                success_rate = metrics.get('success_rate', None)

                if success_rate is not None:
                    is_bad = episode_id in info['bad_eps'] or -1 in info['bad_eps']
                    is_practice = info['practice']
                    is_flawed = info['flawed'] or is_bad

                    # Filter based on flags
                    if not include_practice and is_practice:
                        continue
                    if not include_flawed and is_flawed:
                        continue

                    participant_data[info['name']].append({
                        'success_rate': success_rate,
                        'practice': is_practice,
                        'flawed': is_flawed,
                        'timestamp': ts,
                        'episode_id': episode_id,
                        'crash_rate': metrics.get('crash_rate', 0),
                        'out_of_road_rate': metrics.get('out_of_road_rate', 0),
                        'takeover_rate': metrics.get('takeover_rate', 0),
                    })
            except Exception as e:
                print(f"Error processing {traj_file}: {e}")

    return dict(participant_data)


def print_summary(participant_data: dict[str, list[dict]]) -> None:
    """Print data summary to console."""
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    for name, episodes in sorted(participant_data.items()):
        total = len(episodes)
        avg_success = np.mean([e['success_rate'] for e in episodes]) if episodes else 0
        print(f"{name:12s}: {total:3d} episodes, avg_success={avg_success:.2%}")

    print("=" * 60)


def create_plot(
    participant_data: dict[str, list[dict]],
    output_path: Path,
    title: str = "Participant Success Rates in Driving Study",
) -> None:
    """Create and save the success rate visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    participants = sorted(participant_data.keys())

    # Filter out participants with no data
    participants = [p for p in participants if participant_data[p]]

    if not participants:
        print("No valid participant data to plot!")
        return

    # Calculate stats
    avg_success_rates = []
    std_success_rates = []
    episode_counts = []

    for name in participants:
        episodes = participant_data[name]
        if episodes:
            rates = [e['success_rate'] for e in episodes]
            avg_success_rates.append(np.mean(rates))
            std_success_rates.append(np.std(rates))
            episode_counts.append(len(episodes))
        else:
            avg_success_rates.append(0)
            std_success_rates.append(0)
            episode_counts.append(0)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(participants)))

    # 1. Bar chart: Average success rate per participant
    ax1 = axes[0, 0]
    bars = ax1.bar(
        participants, avg_success_rates, yerr=std_success_rates,
        capsize=3, color=colors, edgecolor='black', linewidth=0.5
    )
    ax1.set_ylabel('Success Rate', fontsize=11)
    ax1.set_xlabel('Participant', fontsize=11)
    ax1.set_title('Average Success Rate', fontsize=12)
    ax1.set_ylim(0, 1.1)

    overall_mean = np.mean(avg_success_rates)
    ax1.axhline(
        y=overall_mean, color='red', linestyle='--',
        label=f'Overall Mean: {overall_mean:.2%}'
    )
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    for bar, count in zip(bars, episode_counts):
        ax1.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'n={count}', ha='center', va='bottom', fontsize=8
        )

    # 2. Box plot: Distribution of success rates
    ax2 = axes[0, 1]
    box_data = []
    box_labels = []
    for name in participants:
        if participant_data[name]:
            box_data.append([e['success_rate'] for e in participant_data[name]])
            box_labels.append(name)

    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax2.set_ylabel('Success Rate', fontsize=11)
    ax2.set_xlabel('Participant', fontsize=11)
    ax2.set_title('Success Rate Distribution', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # 3. Stacked bar: Success vs failure counts
    ax3 = axes[1, 0]
    success_counts = []
    failure_counts = []

    for name in participants:
        episodes = participant_data[name]
        successes = sum(1 for e in episodes if e['success_rate'] == 1.0)
        failures = len(episodes) - successes
        success_counts.append(successes)
        failure_counts.append(failures)

    x = np.arange(len(participants))
    width = 0.6
    ax3.bar(
        x, success_counts, width, label='Success',
        color='#2ecc71', edgecolor='black', linewidth=0.5
    )
    ax3.bar(
        x, failure_counts, width, bottom=success_counts,
        label='Failure', color='#e74c3c', edgecolor='black', linewidth=0.5
    )
    ax3.set_ylabel('Episode Count', fontsize=11)
    ax3.set_xlabel('Participant', fontsize=11)
    ax3.set_title('Episode Outcomes (Success vs Failure)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(participants, rotation=45)
    ax3.legend()

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    all_episodes = []
    for name in participants:
        all_episodes.extend(participant_data[name])

    total_episodes = len(all_episodes)
    total_successes = sum(1 for e in all_episodes if e['success_rate'] == 1.0)
    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0

    summary_text = f"""
STUDY SUMMARY
{'='*40}

Total Participants: {len(participants)}
Total Episodes: {total_episodes}
Total Successes: {total_successes}
Total Failures: {total_episodes - total_successes}
Overall Success Rate: {overall_success_rate:.1%}

INDIVIDUAL STATISTICS
{'='*40}
{'Participant':<12} {'Episodes':>8} {'Success':>8} {'Rate':>8}
{'-'*40}
"""

    # Sort by success rate descending
    sorted_participants = sorted(
        participants,
        key=lambda n: -np.mean([e['success_rate'] for e in participant_data[n]] or [0])
    )

    for name in sorted_participants:
        episodes = participant_data[name]
        n_eps = len(episodes)
        n_success = sum(1 for e in episodes if e['success_rate'] == 1.0)
        rate = n_success / n_eps if n_eps > 0 else 0
        summary_text += f"{name:<12} {n_eps:>8} {n_success:>8} {rate:>7.1%}\n"

    ax4.text(
        0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, bbox_inches='tight',
        facecolor='white', edgecolor='none'
    )
    print(f"Graph saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot participant success rates from driving study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=SCRIPT_DIR / "study_data",
        help="Base directory containing trajectory data",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=SCRIPT_DIR / "study_data" / "traj_metadata.json",
        help="Path to trajectory metadata JSON",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=SCRIPT_DIR / "participant_success_rates.png",
        help="Output path for the PNG file",
    )
    parser.add_argument(
        "--include-practice",
        action="store_true",
        help="Include practice episodes in the analysis",
    )
    parser.add_argument(
        "--include-flawed",
        action="store_true",
        help="Include flawed episodes in the analysis",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Participant Success Rates in Driving Study",
        help="Title for the plot",
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Show real participant names instead of anonymized IDs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    participant_data = load_participant_data(
        base_dir=args.base_dir,
        metadata_path=args.metadata,
        include_practice=args.include_practice,
        include_flawed=args.include_flawed,
    )

    # Anonymize by default
    if not args.no_anonymize:
        participant_data, name_mapping = anonymize_participants(participant_data)
        print("Participant ID mapping (alphabetical):")
        for orig, anon in sorted(name_mapping.items()):
            print(f"  {orig} -> {anon}")
        print()

    print_summary(participant_data)

    create_plot(
        participant_data=participant_data,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
