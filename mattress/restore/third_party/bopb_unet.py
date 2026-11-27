# Scratch / fold detection U-Net from "Bringing Old Photos Back to Life"
# https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life (Global/detection_models)
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Vendored with minimal changes: the multi-GPU sync-batchnorm wrapper was dropped (it was a
# no-op for inference) and the antialiased Downsample layer was inlined.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    # https://github.com/adobe/antialiased-cnns
    def __init__(self, pad_type="reflect", filt_size=3, stride=2, channels=None, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2)),
                          int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [p + pad_off for p in self.pad_sizes]
        self.stride = stride
        self.channels = channels
        a = {1: [1.0], 2: [1.0, 1.0], 3: [1.0, 2.0, 1.0], 4: [1.0, 3.0, 3.0, 1.0],
             5: [1.0, 4.0, 6.0, 4.0, 1.0]}[filt_size]
        a = np.array(a)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp):
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class UNetConvBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding, batch_norm):
        super().__init__()
        block = []
        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
            )
        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, h, w = layer.size()
        dy, dx = (h - target_size[0]) // 2, (w - target_size[1]) // 2
        return layer[:, :, dy: dy + target_size[0], dx: dx + target_size[1]]

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, self.center_crop(bridge, up.shape[2:])], 1)
        return self.conv_block(out)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6, padding=True,
                 batch_norm=True, up_mode="upsample", with_tanh=False, antialiasing=True):
        super().__init__()
        self.depth = depth - 1
        self.first = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 2 ** wf, kernel_size=7),
                                   nn.LeakyReLU(0.2, True))
        prev = 2 ** wf
        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing and depth > 0:
                self.down_sample.append(nn.Sequential(
                    nn.ReflectionPad2d(1), nn.Conv2d(prev, prev, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(prev), nn.LeakyReLU(0.2, True), Downsample(channels=prev, stride=2)))
            else:
                self.down_sample.append(nn.Sequential(
                    nn.ReflectionPad2d(1), nn.Conv2d(prev, prev, kernel_size=4, stride=2, padding=0),
                    nn.BatchNorm2d(prev), nn.LeakyReLU(0.2, True)))
            self.down_path.append(UNetConvBlock(conv_num, prev, 2 ** (wf + i + 1), padding, batch_norm))
            prev = 2 ** (wf + i + 1)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(UNetUpBlock(conv_num, prev, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev = 2 ** (wf + i)
        last = [nn.ReflectionPad2d(1), nn.Conv2d(prev, out_channels, kernel_size=3)]
        if with_tanh:
            last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

    def forward(self, x):
        x = self.first(x)
        blocks = []
        for i, down in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)
