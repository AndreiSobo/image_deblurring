import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='group'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(out_ch)
            self.norm2 = nn.BatchNorm2d(out_ch)
        else:
            # GroupNorm with groups=8 is robust for small batch sizes
            self.norm1 = nn.GroupNorm(8, out_ch)
            self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='group'):
        super().__init__()
        # used bilinear upsample + conv to avoid transpose conv artifacts
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.conv = ConvBlock(in_ch=out_ch*2, out_ch=out_ch, norm=norm)  # after concat
    def forward(self, x, skip):
        x = self.up(x)
        # center-crop skip if needed to match spatial dims (robustness)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DeblurUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, norm='group'):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, norm=norm)
        self.enc2 = ConvBlock(base_ch, base_ch*2, norm=norm)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4, norm=norm)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8, norm=norm)        
        self.pool = nn.MaxPool2d(2,2)

        self.bottleneck = ConvBlock(base_ch*8, base_ch*16, norm=norm)

        self.up4 = UpBlock(base_ch*16, base_ch*8, norm=norm)
        self.up3 = UpBlock(base_ch*8, base_ch*4, norm=norm)
        self.up2 = UpBlock(base_ch*4, base_ch*2, norm=norm)
        self.up1 = UpBlock(base_ch*2, base_ch, norm=norm)

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))
        
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.final(d1)
        return out