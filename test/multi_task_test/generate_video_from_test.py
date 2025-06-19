import argparse
from multi_task_il.datasets.savers import Trajectory
import pickle as pkl
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import cv2 
from PIL import Image, ImageDraw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_folder_path")
    parser.add_argument("--debug", action='store_true', help="Whether or not attach the debugger")
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

        
    trjs= glob.glob(os.path.join(args.trajectory_folder_path, 'traj*.pkl'))
    trjs.sort()
    
    for trj in tqdm(trjs):
        traj_name = trj.split('/')[-1]
        
        with open(trj, 'rb') as f:
            traj = pkl.load(f)

        
        for t in range(len(traj)):
            if t == 0:
                try:
                    img = Image.fromarray(traj[t]['obs']['image'])
                except:
                    img = Image.fromarray(traj[t]['obs']['camera_front_image'])
                videodims = img.size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                FPS = 10
                os.makedirs(os.path.join(args.trajectory_folder_path, "video"), exist_ok=True)
                output_path = os.path.join(args.trajectory_folder_path, "video", traj_name.replace('.pkl', '.mp4'))
                video = cv2.VideoWriter(output_path, fourcc, FPS, videodims)
            
            try:
                img = traj[t]['obs']['image']
            except:
                img = traj[t]['obs']['camera_front_image']
                
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
        video.release()
        print(f"Video saved to {output_path}")
