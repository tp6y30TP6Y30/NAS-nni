import math
import torch
import nni.retiarii.nn.pytorch as nn
from model.splat import SplAtConv2d, DropBlock2D
import torch.nn.functional as F
from nni.retiarii import model_wrapper

@model_wrapper      # this decorator should be put on the out most PyTorch module
class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		self.norm = nn.BatchNorm2d(out_channels)
		self.acti = nn.ReLU(True)

	def forward(self, x):
		x = self.conv(x)
		x = self.norm(x)
		x = self.acti(x)
		return x

@model_wrapper      # this decorator should be put on the out most PyTorch module
class StemConv(nn.Module):
	def __init__(self, in_channels, hidden, out_channels, kernel_size):
		super().__init__()
		self.conv_block1 = ConvBlock(in_channels, hidden, kernel_size, 2, 1)
		self.conv_block2 = ConvBlock(hidden, hidden, kernel_size, 1, 1)
		self.conv_block3 = ConvBlock(hidden, out_channels, kernel_size, 1, 1)
		self.maxpool = nn.MaxPool2d(kernel_size, 2, 1, 1)

	def forward(self, x):
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = self.conv_block3(x)
		x = self.maxpool(x)
		return x

@model_wrapper      # this decorator should be put on the out most PyTorch module
class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=nn.BatchNorm2d, dropblock_prob=0.0, last_gamma=False):
        super().__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@model_wrapper      # this decorator should be put on the out most PyTorch module
class BaseModel(nn.Module):
	def __init__(self):
		super().__init__()
		stem_dim = nn.ValueChoice([32, 64, 128])
		hidden = nn.ValueChoice([32, 64, 128])
		layer1_out_dim = nn.ValueChoice([64, 128, 256])
		layer2_out_dim = nn.ValueChoice([128, 256, 512])
		layer3_out_dim = nn.ValueChoice([256, 512, 1024])
		layer4_out_dim = nn.ValueChoice([512, 1024, 2048])
		self.stem = StemConv(3, hidden, stem_dim, 3)
		self.layer1 = nn.Sequential(
						Bottleneck(stem_dim, layer1_out_dim),
						Bottleneck(layer1_out_dim, layer1_out_dim),
						Bottleneck(layer1_out_dim, layer1_out_dim),
					  )
		self.layer2 = nn.Sequential(
						Bottleneck(layer1_out_dim, layer2_out_dim),
						Bottleneck(layer2_out_dim, layer2_out_dim),
						Bottleneck(layer2_out_dim, layer2_out_dim),
						Bottleneck(layer2_out_dim, layer2_out_dim),
					  )
		self.layer3 = nn.Sequential(
						Bottleneck(layer2_out_dim, layer3_out_dim),
						Bottleneck(layer3_out_dim, layer3_out_dim),
						Bottleneck(layer3_out_dim, layer3_out_dim),
						Bottleneck(layer3_out_dim, layer3_out_dim),
						Bottleneck(layer3_out_dim, layer3_out_dim),
						Bottleneck(layer3_out_dim, layer3_out_dim),
					  )
		self.layer4 = nn.Sequential(
						Bottleneck(layer3_out_dim, layer4_out_dim),
						Bottleneck(layer4_out_dim, layer4_out_dim),
						Bottleneck(layer4_out_dim, layer4_out_dim),
					  )
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(layer4_out_dim, 50)

	def forward(self, x):
		x = self.stem(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = self.fc(x)
		return x

if __name__ == '__main__':
	model = BaseModel()