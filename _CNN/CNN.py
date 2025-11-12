import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.4):
        super(CNNClassifier, self).__init__()

        # === Feature extractor ===
        self.features = nn.Sequential(
            # Block 1 → [32, 112, 112]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            

            # Block 2 → [64, 56, 56]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),  # 🔹 un po' più forte

            # Block 3 → [128, 28, 28]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            

            # Block 4 → [256, 14, 14]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)  
        )

        # 🔹 Calcola automaticamente la dimensione del flatten
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            n_features = self.features(dummy).view(1, -1).shape[1]

        # === Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 🔹 forte dropout fully connected

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
