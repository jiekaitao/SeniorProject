#!/bin/bash
# Job monitor — checks all running jobs and GPU utilization
# Writes to monitor_log.txt

LOG=/blue/cis4914/jietao/DeepPass/monitor_log.txt
echo "========== $(date) ==========" >> $LOG

# Job status
echo "--- JOBS ---" >> $LOG
squeue -u jietao -o '%j|%T|%M|%N|%b' -h >> $LOG 2>&1

# GPU utilization on each node
echo "--- GPU UTIL ---" >> $LOG
for node in $(squeue -u jietao -o '%N' -h 2>/dev/null | sort -u | grep -v "N/A"); do
    util=$(ssh $node nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
    echo "  $node: $util" >> $LOG
done

# Latest log tails
echo "--- LOG TAILS ---" >> $LOG
for f in /blue/cis4914/jietao/DeepPass/results/sbatch_*.log; do
    # Only check files modified in last 2 hours
    if [ -f "$f" ] && find "$f" -mmin -120 2>/dev/null | grep -q .; then
        last=$(tail -1 "$f" 2>/dev/null)
        name=$(basename "$f")
        echo "  $name: $last" >> $LOG
    fi
done

# Check for completed/failed jobs
echo "--- ALERTS ---" >> $LOG
for f in /blue/cis4914/jietao/DeepPass/results/sbatch_*.log; do
    if [ -f "$f" ] && find "$f" -mmin -120 2>/dev/null | grep -q .; then
        if grep -q "COMPLETE" "$f" 2>/dev/null; then
            echo "  DONE: $(basename $f)" >> $LOG
        elif grep -q "Error\|ERROR\|Traceback" "$f" 2>/dev/null; then
            echo "  ERROR: $(basename $f)" >> $LOG
        fi
    fi
done

echo "" >> $LOG
