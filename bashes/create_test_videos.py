"""
Generate demo/trajectory comparison videos from a test-results step folder
(e.g. .../results_pick_place/run_1/step-199), formatted the same way as
utils/analysis/create_video_from_test.py:
  - left side: 2x2 grid of the conditioning demo frames (context*.pkl)
  - right side: the rollout camera frames (traj*.pkl), with the predicted
    bounding box(es) drawn on top
Videos are encoded as H.264 (via an ffmpeg subprocess, since the local
OpenCV build has no built-in h264 writer). For each trajectory the first
and last composed frame are also dumped as PNGs (via PIL) next to the
video, as a quick visual sanity check of the representation.
"""
import argparse
import glob
import json
import os
import pickle
import re
import subprocess

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

NUMBER_RE = re.compile(r"(\d+)")
EVAL_LINE_RE = re.compile(r"Evaluated traj #(\d+), task #(\d+)")

# pick_place real-world object/bin naming + layout (test/multi_task_test/__init__.py
# ENV_OBJECTS, and the `except` fallback in test/multi_task_test/utils.py:get_gt_bb),
# used to recover which object/bin was the target for a given variation id.
OBJ_NAMES = ["greenbox", "yellowbox", "bluebox", "redbox"]
NUM_BINS_PER_OBJECT = 4


def find_number(name):
    match = NUMBER_RE.search(name)
    return int(match.group()) if match else -1


def sort_key(path):
    return find_number(os.path.basename(path))


def find_config(step_path, max_levels=6):
    current = os.path.abspath(step_path)
    for _ in range(max_levels):
        candidate = os.path.join(current, "config.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"Could not find config.yaml within {max_levels} levels above {step_path}")


def load_crop(config_path, task_name):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    task_cfg = config["tasks_cfgs"][task_name]
    return task_cfg.get("crop") or task_cfg["agent_crop"]


def load_project_name(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("project_name")


def parse_variation_log(log_path):
    """Parse 'Evaluated traj #<n>, task #<variation_id>' lines from a test
    srun log, mapping the saved traj{n}/context{n} index to its variation id.
    Some indices may be missing (e.g. lost to concurrent-worker stdout
    interleaving) -- callers must tolerate a partial mapping."""
    variation_map = {}
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            match = EVAL_LINE_RE.search(line)
            if match:
                variation_map[int(match.group(1))] = int(match.group(2))
    return variation_map


def find_variation_log(script_dir, project_name):
    if not project_name:
        return None
    candidate = os.path.join(script_dir, f"test_{project_name}.txt")
    return candidate if os.path.isfile(candidate) else None


def gt_boxes_from_obj_bb(obj_bb, variation_id):
    """Ground-truth [x1,y1,x2,y2] boxes (already in the full camera_front_image
    pixel space, no crop adjustment needed) for the target object and target
    bin of `variation_id`, mirroring retrieve_bb()/get_gt_bb() in
    test/multi_task_test/utils.py for the real-world pick_place task."""
    if obj_bb is None or variation_id is None:
        return []
    camera = obj_bb.get("camera_front", obj_bb)
    obj_name = OBJ_NAMES[variation_id // NUM_BINS_PER_OBJECT]
    bin_name = f"bin_{variation_id % NUM_BINS_PER_OBJECT}"

    boxes = []
    for name in (obj_name, bin_name):
        entry = camera.get(name)
        if entry is None:
            continue
        ul = entry["upper_left_corner"]
        br = entry["bottom_right_corner"]
        # naming in the saved data is mirrored: bottom_right_corner is the
        # top-left point and upper_left_corner is the bottom-right point.
        boxes.append([br[0], br[1], ul[0], ul[1]])
    return boxes


def adjust_bb(bb, crop_params, img_size):
    x1_old, y1_old, x2_old, y2_old = [int(v) for v in bb]
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = img_size
    box_h = img_height - top - crop_params[1]
    box_w = img_width - left - crop_params[3]

    x_scale = 180 / box_w
    y_scale = 100 / box_h
    x1 = int((x1_old / x_scale) + left)
    x2 = int((x2_old / x_scale) + left)
    y1 = int((y1_old / y_scale) + top)
    y2 = int((y2_old / y_scale) + top)
    return [x1, y1, x2, y2]


def context_tensor_to_frames(context_tensor):
    """[1, T, 3, H, W] float tensor in [0, 1] -> list of T uint8 RGB HWC frames."""
    array = (context_tensor * 255).clamp(0, 255).byte().cpu().numpy()
    array = np.transpose(array[0], (0, 2, 3, 1))  # T,H,W,C
    return [array[t] for t in range(array.shape[0])]


def build_demo_grid(context_frames, out_height, out_width, num_cols=2):
    num_rows = (len(context_frames) + num_cols - 1) // num_cols
    rows = []
    for r in range(num_rows):
        row_frames = []
        for c in range(num_cols):
            idx = r * num_cols + c
            if idx < len(context_frames):
                # RGB -> BGR for cv2/ffmpeg
                row_frames.append(context_frames[idx][:, :, ::-1])
        rows.append(cv2.hconcat(row_frames))
    grid = cv2.vconcat(rows)
    return cv2.resize(grid, (out_width, out_height))


class H264VideoWriter:
    def __init__(self, out_path, width, height, fps):
        self.out_path = out_path
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr):
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def process_pair(step_path, video_dir, number, task_name, crop, fps, min_frames, variation_id):
    context_file = os.path.join(step_path, f"context{number}.pkl")
    traj_file = os.path.join(step_path, f"traj{number}.pkl")
    json_file = os.path.join(step_path, f"traj{number}.json")

    with open(context_file, "rb") as f:
        context_data = pickle.load(f)
    with open(traj_file, "rb") as f:
        traj_data = pickle.load(f)

    traj_result = None
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            traj_result = json.load(f)

    if len(traj_data) < min_frames:
        print(f"[skip] traj{number}: only {len(traj_data)} frames (< {min_frames})")
        return

    context_frames = context_tensor_to_frames(context_data)

    first_frame_img = None
    frames = []
    for t, step in enumerate(traj_data):
        obs = step["obs"]
        traj_frame = np.array(obs["camera_front_image"])
        traj_height, traj_width = traj_frame.shape[:2]

        predicted_bb = obs.get("predicted_bb")
        if predicted_bb is not None and t > 0:
            boxes = np.asarray(predicted_bb)
            if boxes.ndim == 1:
                boxes = boxes[None]
            for box in boxes:
                bb = adjust_bb(box, crop, (traj_height, traj_width))
                traj_frame = cv2.rectangle(
                    traj_frame.copy(), (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        gt_bb = obs.get("gt_bb")
        if gt_bb is not None and t > 0:
            # explicit gt_bb already saved (e.g. simulated envs) - crop-space, needs adjust_bb
            boxes = np.asarray(gt_bb)
            if boxes.ndim == 1:
                boxes = boxes[None]
            for box in boxes:
                bb = adjust_bb(box, crop, (traj_height, traj_width))
                traj_frame = cv2.rectangle(
                    traj_frame.copy(), (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
        elif variation_id is not None and t > 0:
            # real-world data: recover gt boxes from obj_bb, already in full-frame pixel space
            for bb in gt_boxes_from_obj_bb(obs.get("obj_bb"), variation_id):
                traj_frame = cv2.rectangle(
                    traj_frame.copy(), (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)

        if t == 0:
            demo_grid = build_demo_grid(context_frames, traj_height, traj_width)

        output_frame = cv2.hconcat([demo_grid, traj_frame])

        label = ""
        if traj_result is not None:
            label = f"avg_iou={traj_result.get('avg_iou', 0):.2f}"
        if variation_id is not None:
            obj_name = OBJ_NAMES[variation_id // NUM_BINS_PER_OBJECT]
            bin_name = f"bin_{variation_id % NUM_BINS_PER_OBJECT}"
            label = f"{label}  target={obj_name}->{bin_name}".strip()
        if label:
            cv2.putText(output_frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 0, 0), 1, cv2.LINE_AA)

        frames.append(output_frame)
        if first_frame_img is None:
            first_frame_img = output_frame

    out_name = f"demo_{number}_traj_{number}"
    out_path = os.path.join(video_dir, f"{out_name}.mp4")
    height, width = frames[0].shape[:2]
    writer = H264VideoWriter(out_path, width, height, fps)
    for frame in frames:
        writer.write(frame)
    writer.close()
    print(f"[ok] wrote {out_path} ({len(frames)} frames)")

    # PIL sanity-check dumps of the first and last composed frame
    Image.fromarray(first_frame_img[:, :, ::-1]).save(
        os.path.join(video_dir, f"{out_name}_first.png"))
    Image.fromarray(frames[-1][:, :, ::-1]).save(
        os.path.join(video_dir, f"{out_name}_last.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_path", type=str,
                         help="Path to a step-N results folder (contains context*.pkl / traj*.pkl)")
    parser.add_argument("--task_name", type=str, default="pick_place")
    parser.add_argument("--config_path", type=str, default=None,
                         help="Defaults to the nearest config.yaml found above step_path")
    parser.add_argument("--fps", type=int, default=None,
                         help="Defaults to 10 for 'Real'/'REAL' paths, else 30")
    parser.add_argument("--min_frames", type=int, default=3,
                         help="Trajectories shorter than this are skipped")
    parser.add_argument("--variation_log", type=str, default=None,
                         help="srun log with 'Evaluated traj #n, task #variation_id' lines, "
                              "used to draw ground-truth (green) object/bin boxes for real-world "
                              "data. Defaults to test_<project_name>.txt next to this script.")
    args = parser.parse_args()

    step_path = os.path.abspath(args.step_path)
    config_path = args.config_path or find_config(step_path)
    crop = load_crop(config_path, args.task_name)
    fps = args.fps
    if fps is None:
        fps = 10 if ("Real" in step_path or "REAL" in step_path) else 30

    video_dir = os.path.join(step_path, "video")
    os.makedirs(video_dir, exist_ok=True)

    context_files = sorted(glob.glob(os.path.join(step_path, "context*.pkl")), key=sort_key)
    numbers = [find_number(os.path.basename(f)) for f in context_files]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_name = load_project_name(config_path)
    variation_log = args.variation_log or find_variation_log(script_dir, project_name)
    variation_map = parse_variation_log(variation_log) if variation_log else {}

    print(f"Config: {config_path}")
    print(f"Crop: {crop}  FPS: {fps}")
    print(f"Found {len(numbers)} demo/trajectory pairs in {step_path}")
    if variation_log:
        known = sum(1 for n in numbers if n in variation_map)
        print(f"Variation log: {variation_log} ({known}/{len(numbers)} trajectories have a known "
              f"target -> green gt boxes drawn for those)")
    else:
        print("No variation log found: gt (green) boxes will be skipped")

    for number in numbers:
        try:
            process_pair(step_path, video_dir, number, args.task_name, crop, fps, args.min_frames,
                         variation_map.get(number))
        except FileNotFoundError as e:
            print(f"[skip] pair {number}: {e}")


if __name__ == "__main__":
    main()
