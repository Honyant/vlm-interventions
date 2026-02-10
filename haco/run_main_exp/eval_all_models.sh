#!/bin/bash
# Evaluation script for all trained VLM intervention models
#
# Usage:
#   ./eval_all_models.sh                    # Evaluate all available models
#   ./eval_all_models.sh --num-episodes 10  # Custom number of episodes
#   ./eval_all_models.sh --dry-run          # Show what would be run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_haco_copy.py"
RESULTS_DIR="${SCRIPT_DIR}/eval_results"

# Default parameters
NUM_EPISODES=5
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Timestamp for this evaluation run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "VLM Intervention Model Evaluation"
echo "=============================================="
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Number of episodes: ${NUM_EPISODES}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "=============================================="
echo ""

# Define all models to evaluate
# Format: experiment_name:checkpoint_path
declare -A MODELS=(
    # Baselines
 #   ["pretrained_original"]="${SCRIPT_DIR}/study_assets/checkpoints/best/policy.pt"
 #   ["intervention_only"]="${CHECKPOINT_DIR}/intervention_only/best/policy.pt"
 #   ["hg_dagger"]="${CHECKPOINT_DIR}/hg_dagger/best/policy.pt"

    # VLM-Corrected variants (will be available after running VLM experiments)
 #   ["vlm_corrected_V_SA"]="${CHECKPOINT_DIR}/vlm_corrected_V_SA/best/policy.pt"
 #   ["vlm_corrected_T_V_SA"]="${CHECKPOINT_DIR}/vlm_corrected_T_V_SA/best/policy.pt"
    ["vlm_corrected_T_V_SA_pre"]="${CHECKPOINT_DIR}/vlm_corrected_T_V_SA_pre/best/policy.pt"

    # VLM Strategy variations
 #   ["vlm_weighting"]="${CHECKPOINT_DIR}/vlm_weighting/best/policy.pt"
 #   ["vlm_filtering"]="${CHECKPOINT_DIR}/vlm_filtering/best/policy.pt"
 #   ["vlm_rejection"]="${CHECKPOINT_DIR}/vlm_rejection/best/policy.pt"
)

# Results summary file
SUMMARY_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"
echo "Evaluation Summary - ${TIMESTAMP}" > "${SUMMARY_FILE}"
echo "==================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Track success/failure
declare -A RESULTS

# Evaluate each model
for model_name in "${!MODELS[@]}"; do
    checkpoint_path="${MODELS[$model_name]}"

    echo ""
    echo "=============================================="
    echo "Evaluating: ${model_name}"
    echo "Checkpoint: ${checkpoint_path}"
    echo "=============================================="

    # Check if checkpoint exists
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "SKIPPED: Checkpoint not found"
        RESULTS["${model_name}"]="SKIPPED (no checkpoint)"
        echo "${model_name}: SKIPPED (checkpoint not found)" >> "${SUMMARY_FILE}"
        continue
    fi

    # Output file for this model
    OUTPUT_FILE="${RESULTS_DIR}/${model_name}_${TIMESTAMP}.txt"

    # Build evaluation command
    EVAL_CMD="python ${EVAL_SCRIPT} --use-pytorch --pytorch-model-path ${checkpoint_path} -n ${NUM_EPISODES}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY RUN] Would run: ${EVAL_CMD}"
        RESULTS["${model_name}"]="DRY RUN"
    else
        echo "Running: ${EVAL_CMD}"
        echo ""

        # Run evaluation and capture output
        if ${EVAL_CMD} 2>&1 | tee "${OUTPUT_FILE}"; then
            echo ""
            echo "SUCCESS: Results saved to ${OUTPUT_FILE}"
            RESULTS["${model_name}"]="SUCCESS"

            # Extract summary stats from output
            echo "" >> "${SUMMARY_FILE}"
            echo "${model_name}:" >> "${SUMMARY_FILE}"
            grep -A 20 "EVALUATION SUMMARY" "${OUTPUT_FILE}" >> "${SUMMARY_FILE}" 2>/dev/null || echo "  (no summary found)" >> "${SUMMARY_FILE}"
        else
            echo ""
            echo "FAILED: Check ${OUTPUT_FILE} for details"
            RESULTS["${model_name}"]="FAILED"
            echo "${model_name}: FAILED" >> "${SUMMARY_FILE}"
        fi
    fi
done

# Print final summary
echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo ""
echo "Results:"
for model_name in "${!RESULTS[@]}"; do
    printf "  %-30s %s\n" "${model_name}:" "${RESULTS[$model_name]}"
done
echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""

# Also save a machine-readable results file
JSON_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.json"
echo "{" > "${JSON_FILE}"
echo "  \"timestamp\": \"${TIMESTAMP}\"," >> "${JSON_FILE}"
echo "  \"num_episodes\": ${NUM_EPISODES}," >> "${JSON_FILE}"
echo "  \"models\": {" >> "${JSON_FILE}"
first=true
for model_name in "${!RESULTS[@]}"; do
    if [[ "${first}" != "true" ]]; then
        echo "," >> "${JSON_FILE}"
    fi
    printf "    \"%s\": \"%s\"" "${model_name}" "${RESULTS[$model_name]}" >> "${JSON_FILE}"
    first=false
done
echo "" >> "${JSON_FILE}"
echo "  }" >> "${JSON_FILE}"
echo "}" >> "${JSON_FILE}"
echo "JSON results saved to: ${JSON_FILE}"
