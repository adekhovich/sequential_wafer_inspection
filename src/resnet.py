'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev. https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torchvision

from torch.autograd import Variable

custom_models = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet101']
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

    
class SupConResNet(nn.Module):
    def __init__(self, model_name, num_classes=8, input_channels=1, feat_dim=128, pretrained=False):
        super(SupConResNet, self).__init__()
        
        encoder = init_resnet(model_name, num_classes=num_classes, 
                              input_channels=1, pretrained=False)
        
        in_features = list(encoder.modules())[-1].in_features
        
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        
        self.fc = nn.Linear(in_features, num_classes)
        self.head = nn.Linear(in_features, feat_dim)


    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)
        projected = F.normalize(self.head(feat), dim=1)
        out = self.fc(feat)
        return out, projected
    

class ResNet(nn.Module):
    def __init__(self, model_name, num_classes=8, input_channels=1, pretrained=False):
        super(ResNet, self).__init__()
        
        encoder = init_resnet(model_name, num_classes=num_classes, 
                              input_channels=1, pretrained=False)
        
        in_features = list(encoder.modules())[-1].in_features
    
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out


def init_resnet(model_name, num_classes=10, input_channels=1, pretrained=False):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(in_features=list(model.modules())[-1].in_features, 
                                           out_features=num_classes, bias=True)
            
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
            
        #in_features = list(model.modules())[-1].in_features
        #model.fc = torch.nn.Linear(in_features=in_features, 
        #                           out_features=num_classes, bias=True)    
            
        if input_channels == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                                
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
            
    return model