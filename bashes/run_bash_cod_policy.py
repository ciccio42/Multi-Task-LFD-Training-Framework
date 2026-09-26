import os
import subprocess
import re
import time
import argparse

BASH_SCRIPT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/real_train_mosaic_target_obj_detector_double_policy.sh"
# Fresh policy run (not a resume of the old Batch24 checkpoint): the policy
# is trained jointly with its own copy of the object-detector weights
# (FREEZE_TARGET_OBJ_DETECTOR=false in the bash script), so resuming that
# old checkpoint would keep training against the OLD detector's weights
# and ignore the newly fixed one entirely. mosaic.target_obj_detector_path
# /_step (set as env-var overridable defaults in the bash script) point
# this fresh run at the corrected-anchor-grid detector instead.
FINETUNE = False
RESUME = False
PROJECT_NAME = "Real-1Task-pick_place-Simulated-Agent-Human-Demonstration-UR5e-Agent-MOSAIC-COD-SKIP-0-5-10-15-FixedDetector"
SAVE_PATH = '/mnt/beegfs/frosa/checkpoint_save_folder/iros_policy_v2'
CHECKPOINT_FOLDER = f"{SAVE_PATH}/{PROJECT_NAME}-Batch24"
RESUME_STEP = 0   # unused on a fresh (FINETUNE=False, RESUME=False) start
DEMO_NAME = 'human_rgb'
MAX_EPOCHS = 1000
AGENT_NAME = 'real_eye_in_hand_ur5e'  # [real_eye_in_hand_ur5e or real_new_ur5e]
USE_WRIST_IMG = False  # True if using eye-in-hand camera, False if using front camera
BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}", f"{AGENT_NAME}", f"{PROJECT_NAME}", f"{USE_WRIST_IMG}"]

def get_highest_epoch(folder):
    highest_epoch = -1
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
        
        # Extract job ID from sbatch output
        job_id = None
        for line in result.stdout.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                break
        
        if job_id is None:
            print("Failed to get job ID from sbatch output")
            return

        print(f"Job {job_id} submitted. Waiting for completion...")
        
        
        # Poll the job status using squeue
        while True:
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error checking job status: {result.stderr}")
                return

            if job_id not in result.stdout:
                print(f"Job {job_id} completed.")
                break

            time.sleep(10)  # Wait for 10 seconds before polling again
        
        
        highest_epoch = get_highest_epoch(os.path.join(SAVE_PATH, CHECKPOINT_FOLDER))
        print(f"Highest epoch reached: {highest_epoch}")
        
          
        RESUME_STEP = highest_epoch
        RESUME=True
        BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}", f"{AGENT_NAME}", f"{PROJECT_NAME}", f"{USE_WRIST_IMG}"]
        
        if highest_epoch >= MAX_EPOCHS or highest_epoch >= MAX_EPOCHS-1:
            print("Reached the maximum number of epochs. Exiting.")
            break
        else:
            print("Restarting the bash script...")
            time.sleep(5)  # Optional: wait for a few seconds before restarting

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

            time.sleep(10)  # Wait for 10 seconds before polling again
        
        highest_epoch = get_highest_epoch(os.path.join(SAVE_PATH, CHECKPOINT_FOLDER))
        print(f"Highest epoch reached: {highest_epoch}")
        RESUME_STEP = highest_epoch
        RESUME=True
        BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}",  f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}", f"{AGENT_NAME}", f"{PROJECT_NAME}", f"{USE_WRIST_IMG}"]
    
    
    run_bash_script()