import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A simple CNN-based model for image classification
    with an auxiliary loss prediction head.
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        # Loss prediction head: predicts a scalar for each sample
        self.loss_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x is [batch_size, 3, 32, 32] for CIFAR-10
        x = self.features(x)  # shape: [batch_size, 128, 8, 8]
        x = x.view(x.size(0), -1)  # flatten -> [batch_size, 8192]

        # Get intermediate features from classifier
        feat = self.classifier[0:2](x)  # first Linear + ReLU
        logits = self.classifier[2](feat)  # final Linear -> [batch_size, num_classes]

        # Predict the classification loss from the same features
        loss_pred = self.loss_predictor(feat).view(-1)  # [batch_size]

        return logits, loss_pred
