"""Model architecture implementations for CurvAL experiments.

Supports ResNet-20 for CIFAR-10 and ResNet-50 for ImageNet-100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """ResNet basic residual block."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, use_batch_norm: bool = True):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if use_batch_norm else nn.Identity()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if use_batch_norm else nn.Identity()
        self.use_batch_norm = use_batch_norm
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            layers = [nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(self.expansion * planes))
            self.shortcut = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture for CIFAR-10."""
    
    def __init__(self, depth: int, num_classes: int = 10, use_batch_norm: bool = True):
        super().__init__()
        assert (depth - 2) % 6 == 0, f"Invalid depth {depth}"
        num_blocks = (depth - 2) // 6
        
        self.in_planes = 16
        self.use_batch_norm = use_batch_norm
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if use_batch_norm else nn.Identity()
        
        # ResNet blocks
        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batch_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck block for ResNet-50."""
    
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_batch_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            layers = [nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels * self.expansion))
            self.shortcut = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 for ImageNet-100."""
    
    def __init__(self, num_classes: int = 100, use_batch_norm: bool = True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(BottleneckBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BottleneckBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 512, 3, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int = 1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.use_batch_norm))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def build_model(model_cfg: DictConfig, device: torch.device) -> nn.Module:
    """Build and return model based on configuration."""
    
    if model_cfg.name == "ResNet-20":
        model = ResNet(
            depth=model_cfg.depth,
            num_classes=model_cfg.num_classes,
            use_batch_norm=model_cfg.use_batch_norm,
        )
    elif model_cfg.name == "ResNet-50":
        model = ResNet50(
            num_classes=model_cfg.num_classes,
            use_batch_norm=model_cfg.use_batch_norm,
        )
    else:
        raise ValueError(f"Unknown model: {model_cfg.name}")
    
    # Move to device
    model = model.to(device)
    
    return model