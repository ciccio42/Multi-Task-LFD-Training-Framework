import os
import subprocess
import re
import time

# Auto-resubmitting orchestrator for train_cond_target_obj_detector_removed_spawn_regions.sh,
# following the same pattern as run_bash_sim_convert_skip051015_policy.py: submit via sbatch,
# poll squeue until the job leaves the queue (7h gpuq walltime cap, crash, or completion), read
# the highest saved model_save-<epoch>.pt, and resubmit with resume=true from there, up to 20
# restarts or until MAX_EPOCHS is reached.
BASH_SCRIPT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/train_cond_target_obj_detector_removed_spawn_regions.sh"
BASHES_DIR = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes"

SAVE_PATH = "/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/spawn_region_experiments"
EXP_NAME = "1Task-pick_place-COD_removed_spawn_regions"
# set_same_n=5 * (16 tasks - 0 skipped) = 80, per train_any.py:61-62's bsize recomputation
# (5, not 7, to match train_keypoint_detection_skip.sh's pick_place SET_SAME_N - the established
# recipe for this real COD/keypoint dataset class, distinct from the CTOD reference's 7).
CHECKPOINT_FOLDER = f"{SAVE_PATH}/{EXP_NAME}-Batch80"
MAX_EPOCHS = 90  # hardcoded in the bash script itself (EPOCH=90), mirrored here as the stop condition.

RESUME_PATH = "/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/spawn_region_experiments/1Task-pick_place-COD_removed_spawn_regions-Batch80"
RESUME_STEP = 6
RESUME = True
BASH_ARGUMENTS = [f"{RESUME_PATH}", f"{RESUME_STEP}", f"{RESUME}"]


def get_highest_epoch(folder):
    highest_epoch = -1
    if not os.path.isdir(folder):
        return highest_epoch
    for file in os.listdir(folder):
        match = re.match(r'model_save-(\d+)\.pt', file)
        if match:
            epoch_number = int(match.group(1))
            if epoch_number > highest_epoch:
                highest_epoch = epoch_number
    return highest_epoch


def run_bash_script():
    global RESUME_PATH, RESUME_STEP, RESUME, BASH_ARGUMENTS
    print(f"Running bash script with arguments: {BASH_ARGUMENTS}", flush=True)
    for i in range(1, 20):
        result = subprocess.run(['sbatch', BASH_SCRIPT] + BASH_ARGUMENTS, cwd=BASHES_DIR, capture_output=True, text=True)
        while result.returncode != 0:
            print(f"Error submitting job: {result.stderr.strip()} -- retrying in 60s", flush=True)
            time.sleep(60)
            result = subprocess.run(['sbatch', BASH_SCRIPT] + BASH_ARGUMENTS, cwd=BASHES_DIR, capture_output=True, text=True)

        job_id = None
        for line in result.stdout.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                break

        if job_id is None:
            print("Failed to get job ID from sbatch output", flush=True)
            return

        print(f"Job {job_id} submitted. Waiting for completion...", flush=True)

        while True:
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error checking job status: {result.stderr}", flush=True)
                return

            if job_id not in result.stdout:
                print(f"Job {job_id} completed.", flush=True)
                break

            time.sleep(60)

        highest_epoch = get_highest_epoch(CHECKPOINT_FOLDER)
        print(f"Highest epoch reached: {highest_epoch}", flush=True)
        if highest_epoch < 0:
            print("No checkpoint found yet -- the job likely crashed before "
                  "saving one. Stopping rather than looping on a broken resume.", flush=True)
            return

        RESUME_PATH = CHECKPOINT_FOLDER
        RESUME_STEP = highest_epoch
        RESUME = True
        BASH_ARGUMENTS = [f"{RESUME_PATH}", f"{RESUME_STEP}", f"{RESUME}"]

        if highest_epoch >= MAX_EPOCHS - 1:
            print("Reached the maximum number of epochs. Exiting.", flush=True)
            break
        else:
            print("Restarting the bash script...", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    run_bash_script()
