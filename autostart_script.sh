#!/bin/bash

# Define your thresholds (in percent)
MEMORY_THRESHOLD=15
UTILIZATION_THRESHOLD=10

# Command to execute when GPU is idle
COMMAND_TO_EXECUTE="bash test_ghostface.sh"

# Function to check GPU status
check_gpu_idle() {
    gpu_stats=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,nounits,noheader | tr -d ',')
    read -r mem_used mem_total gpu_util <<< "$gpu_stats"
    mem_used_percent=$((100 * mem_used / mem_total))

    if (( mem_used_percent < MEMORY_THRESHOLD && gpu_util < UTILIZATION_THRESHOLD )); then
        return 0  # GPU is idle
    else
        return 1  # GPU is busy
    fi
}

# Main loop
while true; do
    if check_gpu_idle; then
        echo "GPU is idle. Executing command..."
        eval "$COMMAND_TO_EXECUTE"
        break
    else
        echo "GPU is busy. Waiting..."
    fi
    sleep 60
done
