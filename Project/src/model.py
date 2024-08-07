# model.py

import torch.nn as nn
from torchvision.models.video import r3d_18

class I3D(nn.Module):
    def __init__(self, num_classes):
        super(I3D, self).__init__()
        self.model = r3d_18(weights="KINETICS400_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)
