# import torch
# import torch.nn as nn

# class MRIClassifierFrozenEncodings(nn.Module):
#     def __init__(self, 
#                  in_channels=256,          # NEW: matches your saved encoding shape
#                  spatial_size=64,          # NEW
#                  proj_dim=128,
#                  num_classes=3):
#         super().__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),   # 64 -> 32

#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),   # 32 -> 16

#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)  # -> [32,1,1]
#         )

#         self.projection = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32, proj_dim),
#             nn.ReLU()
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(proj_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):  
#         cnn_output = self.cnn(x)
#         embedding = self.projection(cnn_output)
#         logits = self.classifier(embedding)
#         return logits, embedding

#cnn_frozen.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MRIClassifierFrozenCNN(nn.Module):
    def __init__(self, 
                 in_channels=256,
                 proj_dim=1408,     # REQUIRED
                 num_classes=3):
        super().__init__()

        # ---- CNN feature extractor ----
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64 → 32

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32 → 16

            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 16 → 8

            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)   # -> [B,2048,1,1]
        )

        # ---- Projection to histo embedding dim ----
        self.projection = nn.Sequential(
            nn.Flatten(),               # 2048
            nn.Linear(2048, proj_dim),  # 2048 → 1408
            nn.ReLU()
        )

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feats = self.cnn(x)    
        embedding = self.projection(feats)   
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return logits, embedding
