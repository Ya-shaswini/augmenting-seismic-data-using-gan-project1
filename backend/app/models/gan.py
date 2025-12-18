import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_length=1024, channels=1):
        super(Generator, self).__init__()
        self.init_size = output_length // 32  # 5 downsamplings implies diving by 2^5 = 32
        self.l1 = nn.Sequential(nn.Linear(noise_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, input_length=1024, channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        # The height and width of downsampled image
        ds_size = input_length // 32
        self.adv_layer = nn.Linear(256 * ds_size, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
