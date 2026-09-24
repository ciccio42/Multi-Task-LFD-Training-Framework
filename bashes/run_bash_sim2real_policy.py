import os
import subprocess
import re
import time
import argparse

# Sim-to-real finetune: warm-start the policy from an already-trained
# SIMULATED double-policy checkpoint, then continue training on REAL data
# into a NEW, separate checkpoint folder. This is a two-phase resume:
#   phase 1 (this file's initial state): FINETUNE=True, RESUME=False,
#     RESUME_PATH points at the OLD sim checkpoint -- warm-starts weights,
#     fresh optimizer/step count.
#   phase 2+ (after the first job): switches to RESUME=True, RESUME_PATH
#     pointing at THIS run's own (new) checkpoint folder -- ordinary
#     self-resume, same as run_bash_cod_policy.py.
# The object-detector submodule is loaded separately via
# mosaic.target_obj_detector_path/_step (env-var overridable in the bash
# script, defaulting to the fixed real detector); since that detector's
# architecture differs in shape from whatever detector the sim checkpoint
# saved internally, the shape-mismatch-tolerant loader in train_utils.py
# will keep the sim-trained policy/action weights while still picking up
# the real detector's own weights for the mismatched submodule params.
BASH_SCRIPT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/real_train_mosaic_target_obj_detector_double_policy_sim2real.sh"

SIM_CHECKPOINT_FOLDER = "/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/iros/1Task-pick_place-Simulated-Agent-Human-Demonstration-UR5e-Agent-MOSAIC-COD-SKIP-0-5-10-15-Batch24"
SIM_CHECKPOINT_STEP = 74  # latest model_save-*.pt actually present there

PROJECT_NAME = "Real-1Task-pick_place-Simulated-Agent-Human-Demonstration-UR5e-Agent-MOSAIC-COD-SKIP-0-5-10-15-Sim2Real"
SAVE_PATH = '/mnt/beegfs/frosa/checkpoint_save_folder/iros_sim2real_policy'
CHECKPOINT_FOLDER = f"{SAVE_PATH}/{PROJECT_NAME}-Batch24"  # this run's own folder, once it exists

FINETUNE = True
RESUME = False
RESUME_PATH = SIM_CHECKPOINT_FOLDER
RESUME_STEP = SIM_CHECKPOINT_STEP
DEMO_NAME = 'human_rgb'
MAX_EPOCHS = 1000
AGENT_NAME = 'real_eye_in_hand_ur5e'  # [real_eye_in_hand_ur5e or real_new_ur5e]
USE_WRIST_IMG = False
BASH_ARGUMENTS = ["pick_place", f"{RESUME_PATH}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}", f"{AGENT_NAME}", f"{PROJECT_NAME}", f"{USE_WRIST_IMG}"]

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
    global RESUME_PATH, RESUME_STEP, FINETUNE, RESUME, BASH_ARGUMENTS
    print(f"Running bash script with arguments: {BASH_ARGUMENTS}")
    for i in range(1, 20):  # Limit to 20 restarts
        result = subprocess.run(['sbatch', BASH_SCRIPT] + BASH_ARGUMENTS, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return

        job_id = None
        for line in result.stdout.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                break

        if job_id is None:
            print("Failed to get job ID from sbatch output")
            return

        print(f"Job {job_id} submitted. Waiting for completion...")

        while True:
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error checking job status: {result.stderr}")
                return

            if job_id not in result.stdout:
                print(f"Job {job_id} completed.")
                break

            time.sleep(10)

        # From here on, always resume from THIS run's own folder, not the
        # original sim checkpoint (phase 1 -> phase 2 switch).
        highest_epoch = get_highest_epoch(CHECKPOINT_FOLDER)
        print(f"Highest epoch reached in {CHECKPOINT_FOLDER}: {highest_epoch}")
        if highest_epoch < 0:
            print("No checkpoint found yet in this run's own folder -- stopping "
                  "rather than looping on a job that produced nothing.")
            return

        RESUME_PATH = CHECKPOINT_FOLDER
        RESUME_STEP = highest_epoch
        FINETUNE = False
        RESUME = True
        BASH_ARGUMENTS = ["pick_place", f"{RESUME_PATH}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}", f"{AGENT_NAME}", f"{PROJECT_NAME}", f"{USE_WRIST_IMG}"]

        if highest_epoch >= MAX_EPOCHS or highest_epoch >= MAX_EPOCHS - 1:
            print("Reached the maximum number of epochs. Exiting.")
            break
        else:
            print("Restarting the bash script...")
            time.sleep(5)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run a bash script with job submission and monitoring.")
    argparser.add_argument("--job_id", type=int, default=0)
    argparser.add_argument("--first_run", action="store_true")
    args = argparser.parse_args()

    if args.first_run:
        job_id = str(args.job_id)
        while True:
            print(f"Checking status of job {job_id}...")
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error checking job status: {result.stderr}")
                exit(1)

            if job_id not in result.stdout:
                print(f"Job {job_id} completed.")
                break

            time.sleep(10)

        highest_epoch = get_highest_epoch(CHECKPOINT_FOLDER)
        print(f"Highest epoch reached: {highest_epoch}")
        if highest_epoch >= 0:
            RESUME_PATH = CHECKPOINT_FOLDER
            RESUME_STEP = highest_epoch
            FINETUNE = False
            RESUME = True
            BASH_ARGUMENTS = ["pick_place", f"{RESUME_PATH}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}", f"{AGENT_NAME}", f"{PROJECT_NAME}", f"{USE_WRIST_IMG}"]

    run_bash_script()
