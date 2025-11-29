import torch
import torch.nn as nn
from torchvision import models


class VideoBranch(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.cnn = base
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.reshape(B, T, -1)
        _, (h, _) = self.lstm(feats)
        return h[-1]


class CSIBranch(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)

    def forward(self, x):
        B, T, F = x.shape
        x = x.permute(0, 2, 1)
        feat = self.cnn(x).permute(0, 2, 1)
        _, (h, _) = self.lstm(feat)
        return h[-1]


class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = VideoBranch()
        self.left  = VideoBranch()
        self.right = VideoBranch()
        self.csi   = CSIBranch()

        fusion_dim = 128 * 4  # 3 video branches + 1 csi branch

        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 20)
        )

    def forward(self, item):
        f1 = self.front(item["front"])
        f2 = self.left(item["left"])
        f3 = self.right(item["right"])
        f4 = self.csi(item["csi"])

        fused = torch.cat([f1, f2, f3, f4], dim=1)
        return self.fc(fused)
