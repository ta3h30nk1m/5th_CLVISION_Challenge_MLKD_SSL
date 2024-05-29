from typing import Callable, List, Type
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models import ResNet
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck

class ResNet_custom(ResNet):
    def __init__(self, block: type[BasicBlock] | type[Bottleneck], layers: List[int], num_classes: int = 1000, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation: List[bool] | None = None, norm_layer: Callable[..., Module] | None = None) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
    
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x, get_feature=False, dropout=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if dropout:
            x = self.fc(self.dropout(feat))
        else:
            x = self.fc(feat)
        if get_feature:
            return x, feat
        return x        

def load_resnet18():
    model = ResNet_custom(BasicBlock, [2,2,2,2], 2)
    return model