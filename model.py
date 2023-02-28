import torch.nn as nn


class Segmentation_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv3d(1, 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),

            nn.Conv3d(128, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Sigmoid(),
        ).double()

    def forward(self, data):
        return self.network(data)