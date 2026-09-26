#!/bin/bash

# Define the pattern to search for (your username)
TARGET="frosa@unisa.it"

echo "Starting to monitor and kill jobs for user: $TARGET"

while true; do
    # Get the list of Job IDs for the user
    # squeue -h removes the header, -u filters by user, -t R filters running
    # We use awk to grab the first column (Job ID)
    JOBS=$(squeue -h -u "$TARGET" | awk '{print $1}')

    if [ -z "$JOBS" ]; then
        echo "No active jobs found. Success!"
        break
    fi

    echo "Found jobs: $JOBS"
    echo "Canceling jobs..."
    
    # scancel is faster and cleaner than manual 'kill' for Slurm jobs
    echo "$JOBS" | xargs scancel

    # Short sleep to prevent CPU hammering and allow the controller to update
    sleep 30
done