import torch
import torch.nn as nn

class BasicBlock_Dilated(nn.Module):
    """(Conv2d_with_Dilation -> BatchNorm -> ReLU)"""
    def __init__(self, in_channels, out_channels, dilation=1, anti_noise=False):
        super().__init__()
        kernel = 5 if anti_noise else 3
        pad = ((kernel - 1) * dilation) // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, 
                      padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ParticleDetectionNet_Passthrough(nn.Module):
    def __init__(self, in_channels=1, num_p=1):
        super(ParticleDetectionNet_Passthrough, self).__init__()
        
        self.input_stage = BasicBlock_Dilated(in_channels, 16, dilation=1, anti_noise=True)

        self.layer1 = BasicBlock_Dilated(16, 32, dilation=2)
        self.layer2 = BasicBlock_Dilated(32, 64, dilation=4)

        self.layer3 = BasicBlock_Dilated(64, 32, dilation=1)
        
        self.outc = nn.Conv2d(32, num_p, kernel_size=1)

    def forward(self, x):
        x = self.input_stage(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.outc(x)
        return torch.sigmoid(out)