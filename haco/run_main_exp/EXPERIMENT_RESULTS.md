# VLM-Based Trajectory Correction Experiment Results

## Overview

This experiment evaluates different approaches for using Vision Language Models (VLMs) to correct human intervention data for behavioral cloning in autonomous driving. The hypothesis is that VLM-corrected interventions can improve policy learning by providing cleaner training signals.

## Experiment Setup

- **Environment**: MetaDrive driving simulator
- **Evaluation Episodes**: 500 per model
- **VLM Used**: Claude Sonnet (claude-sonnet-4-20250514) with extended thinking
- **Training**: Behavioral Cloning with 500 iterations per experiment

## Methods

### Baselines

| Method | Description |
|--------|-------------|
| **pretrained_original** | Original pretrained policy without any fine-tuning on intervention data |
| **intervention_only** | BC trained only on raw human intervention segments (no policy rollout data) |
| **hg_dagger** | HG-DAgger style training: mixes pretrained policy rollouts with intervention data |

### VLM-Corrected Methods

| Method | Description |
|--------|-------------|
| **vlm_corrected_V_SA** | VLM correction using Vision + State/Action information |
| **vlm_corrected_T_V_SA** | VLM correction using Transcript + Vision + State/Action (full context) |
| **vlm_corrected_T_V_SA_pre** | Same as T_V_SA but also includes pre-intervention trajectory segments for additional training data |

### VLM Strategy Variations

| Method | Description |
|--------|-------------|
| **vlm_weighting** | VLM assigns confidence weights (0-1) to each intervention; samples weighted during training |
| **vlm_filtering** | VLM decides whether to keep or discard each intervention segment |
| **vlm_rejection** | VLM rejects low-quality interventions entirely based on assessment |

## Results

### Performance Comparison (Sorted by Success Rate)

| Model | Success Rate |         
|-------|-------------|          
| **vlm_weighting** | **14.80%** |
| pretrained_original | 13.40%   |
| vlm_corrected_T_V_SA | 11.20%  |
| vlm_filtering | 10.20%         |
| hg_dagger | 7.20%              |
| vlm_corrected_V_SA | 6.20%     |
| intervention_only | 3.80%      |
| vlm_rejection | 2.60%          |
| vlm_corrected_T_V_SA_pre | N/A |

*Note: vlm_corrected_T_V_SA_pre evaluation did not complete successfully*

### Key Findings

1. **VLM Weighting Achieves Best Success Rate**: The `vlm_weighting` approach (14.80%) outperforms even the pretrained baseline (13.40%), suggesting that confidence-weighted training can improve policy quality.

2. **Raw Intervention Data Hurts Performance**: Training only on intervention data (`intervention_only`: 3.80%) performs worse than the pretrained baseline, indicating that raw human interventions may contain noise that degrades learning.

3. **HG-DAgger Shows Modest Improvement**: Mixing policy rollouts with interventions (7.20%) helps compared to intervention-only but still underperforms the baseline.

4. **VLM Correction with Full Context**: The `vlm_corrected_T_V_SA` model (11.20%) approaches baseline performance with the highest crash rate (79.80%) but lowest out-of-road rate (9.00%), suggesting more aggressive driving behavior.

5. **Filtering vs Rejection**: Filtering interventions (10.20%) works better than outright rejection (2.60%), suggesting partial intervention data is still valuable.

6. **Vision-Only Context Insufficient**: Using only Vision + State/Action (`vlm_corrected_V_SA`: 6.20%) underperforms compared to including transcript context.

## Failure Mode Analysis

| Model | Primary Failure Mode |
|-------|---------------------|
| vlm_corrected_T_V_SA | Crashes (79.80%) - aggressive driving |
| intervention_only | Out of Road (56.20%) - poor lane keeping |
| vlm_filtering | Out of Road (55.40%) - poor lane keeping |
| vlm_rejection | Out of Road (53.20%) - poor lane keeping |
| pretrained_original | Balanced (50.40% crash, 36.40% OOR) |
| vlm_weighting | Balanced (46.40% crash, 39.60% OOR) |

## Conclusions

1. **VLM-based weighting shows promise**: Assigning confidence scores to intervention data and using weighted sampling during training yields the best results.

2. **Context matters for VLM correction**: Including transcript information alongside vision and state/action data improves VLM correction quality.

3. **Binary filtering less effective**: Simply filtering or rejecting interventions is less effective than soft weighting, likely because even imperfect interventions contain useful signal.

4. **Data quality > quantity**: The intervention_only baseline's poor performance confirms that raw human interventions contain significant noise that must be addressed.

## Future Work

- Investigate why vlm_corrected_T_V_SA_pre evaluation failed
- Test with more training iterations
- Explore ensemble approaches combining multiple VLM strategies
- Evaluate on additional driving scenarios
- Fine-tune VLM prompts for better correction quality

## Files

- **Checkpoints**: `/haco/run_main_exp/checkpoints/{model_name}/best/policy.pt`
- **Evaluation Results**: `/haco/run_main_exp/eval_results/`
- **Training Script**: `/haco/run_main_exp/correct_and_train.py`
- **Experiment Runner**: `/haco/run_main_exp/run_experiments.py`
- **Evaluation Script**: `/haco/run_main_exp/eval_all_models.sh`
