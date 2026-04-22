#!/bin/bash
# GPU Ping: exits (triggering Claude notification) when GPU count drops below expected
# Run as: bash gpu_ping.sh <expected_count>
# When this exits, Claude gets a task notification and can react

EXPECTED=${1:-5}
CHECK_INTERVAL=60
MAX_CHECKS=30  # 30 minutes max before re-ping anyway

for i in $(seq 1 $MAX_CHECKS); do
    CURRENT=$(squeue -u $USER -o '%j|%T' -h 2>/dev/null | grep -v bash | grep -c "|")

    if [ "$CURRENT" -lt "$EXPECTED" ]; then
        echo "GPU_DROP: $CURRENT/$EXPECTED jobs running"
        echo "Current jobs:"
        squeue -u $USER -o '%j|%T|%M' -h 2>/dev/null | grep -v bash
        echo "---"
        # Check what finished/crashed
        for f in /blue/cis4914/jietao/DeepPass/results/sbatch_*.log; do
            if [ -f "$f" ] && find "$f" -mmin -5 2>/dev/null | grep -q .; then
                name=$(basename "$f")
                if grep -q "COMPLETE\|Finished" "$f" 2>/dev/null; then
                    echo "DONE: $name"
                elif grep -q "Error\|Traceback" "$f" 2>/dev/null; then
                    echo "CRASH: $name"
                fi
            fi
        done
        exit 0  # This triggers Claude's notification
    fi

    sleep $CHECK_INTERVAL
done

# Max time reached — report status anyway
echo "HEARTBEAT: $CURRENT/$EXPECTED jobs after $MAX_CHECKS checks"
squeue -u $USER -o '%j|%T|%M' -h 2>/dev/null | grep -v bash
exit 0
