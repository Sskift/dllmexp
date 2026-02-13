#!/bin/bash
# Show progress of exp.sh experiments.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$REPO_ROOT/exp_state"
LOG_DIR="$REPO_ROOT/exp_logs"
STATUS_LOG="$STATE_DIR/status.log"
TOTAL=48

count_meta() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -maxdepth 1 -type f -name '*.meta' | wc -l
    else
        echo 0
    fi
}

running_count=$(count_meta "$STATE_DIR/running")
done_count=$(count_meta "$STATE_DIR/done")
failed_count=$(count_meta "$STATE_DIR/failed")
completed=$((done_count + failed_count))
if (( TOTAL > 0 )); then
    percent=$(awk -v c="$completed" -v t="$TOTAL" 'BEGIN{printf "%.1f", (c*100)/t}')
else
    percent="0.0"
fi

echo "====== 实验进度 ======"
echo "运行中: $running_count | 已完成: $done_count | 失败: $failed_count | 总进度: $completed/$TOTAL (${percent}%)"

echo "当前运行任务:" 
if [[ "$running_count" -eq 0 ]]; then
    echo "- 无"
else
    for f in "$STATE_DIR/running"/*.meta; do
        [[ -e "$f" ]] || continue
        model=$(grep '^model=' "$f" | cut -d= -f2)
        task=$(grep '^task=' "$f" | cut -d= -f2)
        tps=$(grep '^tps=' "$f" | cut -d= -f2)
        gpu=$(grep '^gpu=' "$f" | cut -d= -f2)
        start_ts=$(grep '^start_ts=' "$f" | cut -d= -f2)
        if [[ -n "$start_ts" ]]; then
            now=$(date +%s)
            elapsed=$((now - start_ts))
            # format elapsed as H:MM:SS
            h=$((elapsed/3600))
            m=$(((elapsed%3600)/60))
            s=$((elapsed%60))
            printf -- "- %s | %s | tps=%s | gpu=%s | 进度(已运行): %d:%02d:%02d\n" "$model" "$task" "$tps" "$gpu" "$h" "$m" "$s"
        else
            echo "- ${model} | ${task} | tps=${tps} | gpu=${gpu} | 进度: 未知"
        fi
    done
fi

if [[ "$failed_count" -gt 0 ]]; then
    echo "最近失败任务:"
    # List last 3 failed
    grep "FAIL" "$STATUS_LOG" | tail -n 3 | while read -r line; do
        echo "- $line"
    done
fi

echo "GPU 状态:" 
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU%u %s %s/%s MiB util %s%%\n", $1,$2,$3,$4,$5}'
else
    echo "nvidia-smi 未找到"
fi

echo "最近 3 条状态日志:" 
if [[ -f "$STATUS_LOG" ]]; then
    tail -n 3 "$STATUS_LOG"
else
    echo "状态日志不存在"
fi
