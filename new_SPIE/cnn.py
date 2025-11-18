#cnn.py
import os
import csv
import shutil
import argparse
import random
import logging
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

class MRIClassifierCNN(nn.Module):
    def __init__(self,
                 sam_model,                 # loaded SAM/MedSAM model
                 num_classes: int = 3,
                 proj_dim: int = 128,
                 use_pre_neck: bool = True):
        super().__init__()
        self.encoder = sam_model.image_encoder
        stem = []
        if use_pre_neck and hasattr(self.encoder, "neck"):
            self.encoder.neck = nn.Identity()
            stem = [
                nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ]
        self.cnn = nn.Sequential(
            *stem,
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, proj_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        medSAM_output = self.encoder(x)
        cnn_output = self.cnn(medSAM_output)
        embedding = self.projection(cnn_output)
        logits = self.classifier(embedding)
        return logits, embedding




