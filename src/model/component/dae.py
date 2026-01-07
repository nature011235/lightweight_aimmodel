import torch.nn as nn

from .encoder import Encoder


class MobileNetDAE(nn.Module):
    def __init__(self):
        super(MobileNetDAE, self).__init__()

        self.encoder = Encoder(pretrained=False, frozen=False)
        # output [b,576,h/32,w/32]

        self.decoder = nn.Sequential(
            # Block 1: 8x8 -> 16x16
            nn.ConvTranspose2d(
                576, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 2: 16x16 -> 32x32
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 3: 32x32 -> 64x64
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 4: 64x64 -> 128x128
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 5: 128x128 -> 256x256
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # output 0~1
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstructed = self.decoder(features)

        return reconstructed
