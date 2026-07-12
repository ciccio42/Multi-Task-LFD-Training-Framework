import os
import subprocess
import re
import time
import argparse

BASH_SCRIPT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/real_train_mosaic_target_obj_detector_double_policy.sh" #"/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/train_mosaic_target_obj_detector_double_policy.sh"
FINETUNE = False
RESUME = True
CHECKPOINT_FOLDER = "/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/iros/Real-1Task-pick_place-Simulated-Agent-Human-Demonstration-UR5e-Agent-MOSAIC-COD-SKIP-0-5-10-15-EYE-IN-HAND--Batch24"
RESUME_STEP = 8
DEMO_NAME = 'human_rgb'
SAVE_PATH = '/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/iros'
MAX_EPOCHS = 1000  # Set your maximum number of epochs here
BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}"]

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
        BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}", f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}"]
        
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
        BASH_ARGUMENTS = ["pick_place", f"{CHECKPOINT_FOLDER}", f"{RESUME_STEP}", f"{FINETUNE}", f"{RESUME}",  f"{DEMO_NAME}", f"{SAVE_PATH}", f"{MAX_EPOCHS}"]
    
    
    run_bash_script()