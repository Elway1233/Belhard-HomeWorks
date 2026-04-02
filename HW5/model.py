import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

LATENT_DIM = 512


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return torch.nn.functional.leaky_relu(x + self.block(x), 0.2, True)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            ResBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            ResBlock(256),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Flatten()
        )
        self.fc = nn.Linear(512 * 4 * 4, LATENT_DIM)

    def forward(self, x):
        features = self.model(x)
        return self.fc(features)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 512 * 4 * 4)
        self.model = nn.Sequential(
            nn.BatchNorm2d(512),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            ResBlock(256),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            ResBlock(128),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
