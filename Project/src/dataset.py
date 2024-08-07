# dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils import count_frames, create_clip_indices

# Define constants
DATA_DIR = 'data/raw_frames'
IMG_HEIGHT = 150
IMG_WIDTH = 150
MAX_FRAMES = 450

class VideoDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.frame_counts = count_frames(self.annotations)
        self.clip_indices = create_clip_indices(self.annotations)

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        start_idx = self.clip_indices[idx]
        row = self.annotations.iloc[start_idx]
        video_name = row['Video Name']
        cow_id = row['Cow ID']
        keyframe = int(row['Position (Seconds)'])
        x1, y1, x2, y2 = (row['Bounding Box (x1)'], row['Bounding Box (y1)'], row['Bounding Box (x2)'], row['Bounding Box (y2)'])
        label = int(row['Behavior Category']) - 2

        num_frames = self.frame_counts.get((video_name, cow_id), 0)
        frames = self.load_frames(video_name, num_frames, x1, y1, x2, y2)

        # Ensure frames have the same length
        frames = list(frames)
        if len(frames) < MAX_FRAMES:
            frames.extend([torch.zeros((3, IMG_HEIGHT, IMG_WIDTH))] * (MAX_FRAMES - len(frames)))
        elif len(frames) > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]

        clip = torch.stack(frames)

        return clip, label

    def load_frames(self, video_name, num_frames, x1, y1, x2, y2):
        for i in range(MAX_FRAMES):
            frame_path = os.path.join(DATA_DIR, video_name, f'img_{i+1:05d}.jpg')
            if i < num_frames and os.path.exists(frame_path):
                image = read_image(frame_path).float() / 255.0
                _, img_height, img_width = image.shape
                x1_pixel, y1_pixel, x2_pixel, y2_pixel = self.normalize_bounding_box(x1, y1, x2, y2, img_width, img_height)
                cropped_image = image[:, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                if cropped_image.shape[1] > 0 and cropped_image.shape[2] > 0:
                    if self.transform:
                        cropped_image = self.transform(cropped_image)
                    yield cropped_image
                else:
                    yield torch.zeros((3, IMG_HEIGHT, IMG_WIDTH))
            else:
                yield torch.zeros((3, IMG_HEIGHT, IMG_WIDTH))

    def normalize_bounding_box(self, x1, y1, x2, y2, img_width, img_height):
        x1_pixel = max(0, min(int(x1 * img_width), img_width - 1))
        y1_pixel = max(0, min(int(y1 * img_height), img_height - 1))
        x2_pixel = max(x1_pixel + 1, min(int(x2 * img_width), img_width))
        y2_pixel = max(y1_pixel + 1, min(int(y2 * img_height), img_height))
        return x1_pixel, y1_pixel, x2_pixel, y2_pixel
