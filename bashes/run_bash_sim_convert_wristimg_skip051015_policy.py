import os
import subprocess
import re
import time
import argparse

# Simulated pick_place double-policy training, with convert_action=true
# (aligns sim actions with the real robot's base_link frame / gripper
# convention), use_wrist_img=true, and tasks_cfgs=7_tasks_sim_skip_0_5_10_15
# (holds out subtasks 0/5/10/15, matching the real-world policy's split).
# Meant as a sim baseline for a future sim-to-real finetune on the real
# eye-in-hand data. The sim ur5e_pick_place dataset didn't carry
# eye_in_hand_image before -- see add_eye_in_hand_sim_dataset.py, which
# extended opt_dataset/ur5e_pick_place with it from the raw no_opt_dataset
# trajectories. Fresh start: this EXP_NAME/checkpoint folder doesn't exist
# yet, so RESUME_FOLDER/STEP on the first submission are unused
# (FINETUNE=False, RESUME=False).
#
# The gpuq partition caps jobs at 7h wall time -- run_bash_script() below
# already auto-resumes past that: it waits for the SLURM job to leave the
# queue (whether from the 7h cap, a crash, or completion), reads the
# highest saved checkpoint epoch, and resubmits with RESUME=true from
# there, up to 20 restarts or until MAX_EPOCHS is reached.
BASH_SCRIPT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/train_mosaic_target_obj_detector_double_policy_convert_wristimg_skip051015.sh"

FINETUNE = False
RESUME = False
# SAVE_PATH must be the shared iros checkpoint root: the script's own
# TARGET_OBJ_DETECTOR_PATH for pick_place is `${SAVE_PATH}/1Task-pick_place-
# ...-COD-SKIP-0-5-10-15-Batch60` (relative to SAVE_PATH), which only
# resolves correctly if SAVE_PATH is this shared root where that sim
# detector checkpoint actually lives.
SAVE_PATH = '/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/iros'
PROJECT_NAME = "1Task-pick_place-Simulated-Agent-Human-Demonstration-UR5e-Agent-MOSAIC-COD-SKIP-0-5-10-15-ConvertAction-WristImg"
CHECKPOINT_FOLDER = f"{SAVE_PATH}/{PROJECT_NAME}-Batch24"
RESUME_STEP = 0   # unused on a fresh (FINETUNE=False, RESUME=False) start
DEMO_NAME = 'human_rgb'
USE_WRIST_IMG = True
# EPOCH is hardcoded to 90 inside the bash script itself (no CLI override
# for this sim script, unlike the real_train_* ones) -- mirrored here only
# for the orchestrator's own stop condition.
MAX_EPOCHS = 90
BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{USE_WRIST_IMG}"]

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
    global RESUME_STEP, RESUME, BASH_ARGUMENTS
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

        highest_epoch = get_highest_epoch(CHECKPOINT_FOLDER)
        print(f"Highest epoch reached: {highest_epoch}")
        if highest_epoch < 0:
            print("No checkpoint found yet -- the job likely crashed before "
                  "saving one. Stopping rather than looping on a broken resume.")
            return

        RESUME_STEP = highest_epoch
        RESUME = True
        BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{USE_WRIST_IMG}"]

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
        RESUME_STEP = highest_epoch
        RESUME = True
        BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{USE_WRIST_IMG}"]

    run_bash_script()
