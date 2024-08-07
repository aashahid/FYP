# utils.py

import os
import cv2
import torch
import numpy as np

DATA_DIR = 'data/raw_frames'
IMG_HEIGHT, IMG_WIDTH = 150, 150
MAX_FRAMES = 450  # Maximum number of frames per video clip

def count_frames(annotations):
    frame_counts = {}
    for _, row in annotations.iterrows():
        video_name, cow_id = row['Video Name'], row['Cow ID']
        if (video_name, cow_id) not in frame_counts:
            frame_counts[(video_name, cow_id)] = 0
        frame_counts[(video_name, cow_id)] += 1
    return frame_counts

def create_clip_indices(annotations):
    clip_indices = []
    current_video_name, current_cow_id = None, None
    for idx, row in annotations.iterrows():
        video_name, cow_id = row['Video Name'], row['Cow ID']
        if video_name != current_video_name or cow_id != current_cow_id:
            clip_indices.append(idx)
            current_video_name, current_cow_id = video_name, cow_id
    return clip_indices

def save_clip_as_frames(clip, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(clip):
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frame_path = os.path.join(output_dir, f'frame_{i+1:04d}.jpg')
        cv2.imwrite(frame_path, frame_np)
    print(f"Clip frames saved to {output_dir}")
