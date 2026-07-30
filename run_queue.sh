#!/bin/bash
# Tiny GPU job queue: dispatch the jobs listed in $QUEUE_FILE onto the given GPUs,
# one job per GPU at a time.
#
#   bash run_queue.sh <queue_file> <gpu> [<gpu> ...]
set -u
cd "$(dirname "$0")"

queue=$1
shift
gpus=("$@")
lock="$queue.lock"
mkdir -p results/_logs
: > "$lock"

worker() {
    local gpu=$1
    while true; do
        local job
        job=$(flock "$lock" -c "head -n 1 '$queue'; sed -i '1d' '$queue'")
        [ -z "$job" ] && break
        echo "[gpu $gpu] >>> $job"
        bash run_new_comparisons.sh "$job" "$gpu" > "results/_logs/$job.log" 2>&1
        echo "[gpu $gpu] <<< $job (exit $?)"
    done
    echo "[gpu $gpu] queue empty, worker done"
}

for gpu in "${gpus[@]}"; do
    worker "$gpu" &
done
wait
echo "ALL QUEUED JOBS DONE"
