import torch
from torch import nn
from models.base_model import BaseModel


class UNet(BaseModel):
    def __init__(self, in_channels=1, out_channels=2, features=(64, 128, 256)):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Encoder
        self.enc1 = conv_block(in_channels, features[0])
        self.enc2 = conv_block(features[0], features[1])
        self.enc3 = conv_block(features[1], features[2])

        # Decoder
        self.dec2 = conv_block(features[2] + features[1], features[1])
        self.dec1 = conv_block(features[1] + features[0], features[0])

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return self.final(d1)
