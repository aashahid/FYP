# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm

from dataset import VideoDataset
from model import I3D
from utils import save_clip_as_frames

DATA_DIR = 'data/raw_frames'
ANNOTATIONS_DIR = 'data/cvb_in_ava_format'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
IMG_HEIGHT, IMG_WIDTH = 150, 150
NUM_CLASSES = 11
BATCH_SIZE = 16
EPOCHS = 10
MODEL_SAVE_PATH = 'i3d_model.pth'
SAMPLE_FRAMES_DIR = 'sample_frames'

def load_datasets():
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, TRAIN_CSV), transform=transform)
    val_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, VAL_CSV), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=18, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=18, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, train_dataset, val_dataset = load_datasets()

    #num_clips = train_dataset.get_num_clips()
    #print(f'Number of clips in the train dataset: {num_clips}')

    clip, label = train_dataset[0]
    save_clip_as_frames(clip, SAMPLE_FRAMES_DIR)

    model = I3D(num_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == "__main__":
    main()
