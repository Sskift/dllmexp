#!/bin/bash
# Launch 48 diffusion experiments in a tmux session with tokens_per_step sweep.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$REPO_ROOT/exp_state"
LOG_DIR="$REPO_ROOT/exp_logs"
STATUS_LOG="$STATE_DIR/status.log"
SESSION_NAME=${SESSION_NAME:-exp48}
# As per request, limiting to 6,7
GPUS_DEFAULT="6 7"
RETRY_MAX=${RETRY_MAX:-1}
MIN_FREE_MIB=${MIN_FREE_MIB:-20000} # 20GB free required PER JOB
# Allow up to 2 jobs per GPU
MAX_JOBS_PER_GPU=2

MODELS=(llada llada1.5 diffucoder dream)
TASKS=(humaneval mbpp gsm8k truthfulqa)
TPS_VALUES=(1 2 4)

# Internal runner executed inside tmux
run_internal_waitable() {
    set -o pipefail
    mkdir -p "$STATE_DIR"/running "$STATE_DIR"/done "$STATE_DIR"/failed "$LOG_DIR"
    echo "$(date -Ins) RUN_START" >> "$STATUS_LOG"

    # shellcheck disable=SC1090
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    fi
    conda activate contrastive || echo "Warning: Conda env 'contrastive' activation failed or not found."
    
    export HF_HOME=xxx
    export HF_DATASETS_CACHE=xxx
    # Use HOME instead of hardcoded user path where possible
    export LD_LIBRARY_PATH=${HOME}/miniconda3/envs/contrastive/lib:${LD_LIBRARY_PATH:-}
    cd "$REPO_ROOT" || exit 1

    IFS=' ' read -r -a GPUS <<< "${GPUS:-$GPUS_DEFAULT}"
    GPU_COUNT=${#GPUS[@]}
    if (( GPU_COUNT == 0 )); then
        echo "No GPUs specified; aborting." | tee -a "$STATUS_LOG"
        exit 1
    fi
    # Ensure all selected GPUs are visible to subprocesses
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"

    default_alg() {
        case "$1" in
            dream|diffucoder) echo "maskgit_plus" ;;
            llada|llada1.5) echo "low_confidence" ;;
            *) echo "unknown" ;;
        esac
    }

    jobs=()
    for m in "${MODELS[@]}"; do
        for t in "${TASKS[@]}"; do
            for k in "${TPS_VALUES[@]}"; do
                alg=$(default_alg "$m")
                # Expected output filename construction mirroring eval.py
                fname="${m}_${t}_${alg}_tps=${k}_limitNone.json"
                result_path="$REPO_ROOT/results/$fname"
                
                if [[ -f "$result_path" ]]; then
                    echo "Skipping completed job: $fname" | tee -a "$STATUS_LOG"
                    continue
                fi
                
                jobs+=("${m}|${t}|${k}|0")
            done
        done
    done

    # Map job PIDs to job info
    declare -A PID_JOB
    declare -A PID_GPU
    declare -A PID_META
    # Track NUMBER of jobs per GPU, not just binary busy state
    declare -A GPU_JOB_COUNT
    for g in "${GPUS[@]}"; do GPU_JOB_COUNT[$g]=0; done

    gpu_free_mib() {
        local gpu="$1"
        # Only query the specific GPU ID relative to the visible devices or system?
        # nvidia-smi IDs match CUDA_VISIBLE_DEVICES only if they are the same indices.
        # Assuming GPUS contains physical IDs. 
        nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1
    }

    clear_gpu_processes() {
        local gpu="$1"
        # Kill all compute processes on this GPU that are not our children (best-effort)
        # Warning: This is aggressive.
        pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        for p in $pids; do
            if kill -0 "$p" 2>/dev/null; then
                # Avoid killing self or parents (simple heuristic)
                if [[ "$p" != "$$" && "$p" != "$BASHPID" ]]; then
                   kill -9 "$p" 2>/dev/null || true
                fi
            fi
        done
    }

    enqueue_retry() {
        local job="$1" attempts="$2"
        if (( attempts < RETRY_MAX )); then
            local next_attempt=$((attempts + 1))
            jobs+=("${job%|*}|${next_attempt}")
            echo "$(date -Ins) RETRY_ENQUEUE job=${job%|*} attempt=$next_attempt" >> "$STATUS_LOG"
        else
            echo "$(date -Ins) RETRY_SKIP job=${job%|*} attempts=$attempts" >> "$STATUS_LOG"
        fi
    }

    start_job() {
        local job="$1" gpu="$2"
        IFS='|' read -r model task tps attempts <<< "$job"
        local alg
        alg=$(default_alg "$model")
        local job_id="${model}_${task}_tps${tps}"
        local log_file="$LOG_DIR/${job_id}.log"
        local meta_file="$STATE_DIR/running/${job_id}.meta"
        local start_ts
        start_ts=$(date +%s)

        {
            echo "job_id=$job_id"
            echo "model=$model"
            echo "task=$task"
            echo "tps=$tps"
            echo "alg=$alg"
            echo "gpu=$gpu"
            echo "log=$log_file"
            echo "start_ts=$start_ts"
            echo "start_iso=$(date -Ins)"
        } > "$meta_file"
        
        echo "$(date -Ins) START job=$job_id gpu=$gpu log=$log_file" >> "$STATUS_LOG"
        
        # Launch actual experiment
        # Using CUDA_VISIBLE_DEVICES=$gpu to isolate
        CUDA_VISIBLE_DEVICES="$gpu" python eval.py --model_alias "$model" --task "$task" --alg "$alg" --tokens_per_step "$tps" >> "$log_file" 2>&1 &
        local pid=$!
        PID_JOB[$pid]="$job"
        PID_GPU[$pid]="$gpu"
        PID_META[$pid]="$meta_file"
        
        # Increment job count for this GPU
        current_count=${GPU_JOB_COUNT[$gpu]}
        GPU_JOB_COUNT[$gpu]=$((current_count + 1))
    }

    next_free_gpu() {
        for g in "${GPUS[@]}"; do
            # Check slot based availability
            count=${GPU_JOB_COUNT[$g]}
            if (( count < MAX_JOBS_PER_GPU )); then
                # Check actual memory availability
                # We need MIN_FREE_MIB for the *next* job
                local free_mib
                free_mib=$(gpu_free_mib "$g")
                if [[ -n "$free_mib" && "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
                    echo "$g"
                    return 0
                fi
            fi
        done
        return 1
    }

    pending_index=0
    
    while (( pending_index < ${#jobs[@]} || ${#PID_JOB[@]} > 0 )); do
        # Schedule as many jobs as possible
        while (( pending_index < ${#jobs[@]} )); do
            # Try to find a free GPU for the next job
            # Note: We need to limit total dispatched jobs to avoid overwhelming
            # even if next_free_gpu returns true (which it shouldn't if full).
            # The GPU_JOB_COUNT handles the limit.
            
            gpu=$(next_free_gpu)
            if [[ -z "$gpu" ]]; then break; fi
            
            job=${jobs[$pending_index]}
            start_job "$job" "$gpu"
            ((pending_index++))
        done

        if (( ${#PID_JOB[@]} == 0 )); then
            sleep 1
            continue
        fi

        # Wait for any job to finish
        wait -n
        rc=$?
        
        # Identify which PID finished
        finished_pid=""
        for pid in "${!PID_JOB[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                finished_pid="$pid"
                break
            fi
        done
        
        if [[ -z "$finished_pid" ]]; then
             # Race condition or obscure signal; loop again
             sleep 0.5
             continue
        fi

        job=${PID_JOB[$finished_pid]}
        meta=${PID_META[$finished_pid]}
        gpu=${PID_GPU[$finished_pid]}
        
        unset PID_JOB[$finished_pid]
        unset PID_META[$finished_pid]
        unset PID_GPU[$finished_pid]
        
        # Decrement job count
        current_count=${GPU_JOB_COUNT[$gpu]}
        GPU_JOB_COUNT[$gpu]=$((current_count - 1))
        
        IFS='|' read -r model task tps attempts <<< "$job"
        job_id="${model}_${task}_tps${tps}"
        end_ts=$(date +%s)
        start_ts_val=$(grep '^start_ts=' "$meta" | cut -d= -f2)
        if [[ -z "$start_ts_val" ]]; then start_ts_val=$end_ts; fi
        duration=$((end_ts - start_ts_val))
        
        dest_dir="$STATE_DIR/done"
        status_label="DONE"
        if (( rc != 0 )); then
            dest_dir="$STATE_DIR/failed"
            status_label="FAIL"
        fi
        
        dest_meta="$dest_dir/${job_id}.meta"
        {
            if [[ -f "$meta" ]]; then cat "$meta"; fi
            echo "end_ts=$end_ts"
            echo "end_iso=$(date -Ins)"
            echo "rc=$rc"
            echo "duration_sec=$duration"
            echo "attempts=$attempts"
        } > "$dest_meta"
        rm -f "$meta"
        
        echo "$(date -Ins) ${status_label} job=$job_id gpu=$gpu rc=$rc duration_sec=$duration" >> "$STATUS_LOG"
        
        if (( rc != 0 )); then
            enqueue_retry "$job" "$attempts"
        fi
    done

    echo "$(date -Ins) RUN_END" >> "$STATUS_LOG"
}

if [[ "${1:-}" == "--run-internal" ]]; then
    run_internal_waitable
    exit 0
fi

# Outer launcher: start tmux session so jobs survive terminal closure
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "Attach: tmux attach -t $SESSION_NAME"
    exit 0
fi

if ! command -v tmux &> /dev/null; then
    echo "tmux is not installed or not in PATH. Running directly attached..."
    run_internal_waitable
    exit $?
fi

mkdir -p "$STATE_DIR" "$LOG_DIR"
TMUX_LOG="$STATE_DIR/tmux.log"

env_prefix=""
if [[ -n "${GPUS:-}" ]]; then
    env_prefix="GPUS=\"${GPUS}\" "
fi

# Re-invoke self inside tmux
env_cmd="${env_prefix}$SCRIPT_DIR/exp.sh --run-internal"

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME" "bash -lc '$env_cmd' > '$TMUX_LOG' 2>&1" C-m

echo "Started 48 experiments in tmux session '$SESSION_NAME'."
echo "Monitor: bash $SCRIPT_DIR/progress.sh"
echo "Attach:  tmux attach -t $SESSION_NAME"
echo "Log:     $TMUX_LOG"
