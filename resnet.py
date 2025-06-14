import torchvision
from argparse import Namespace
from config import get_config
from typing import Any
import torch
import torch.nn as nn

class _resnet(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self, model):
        super(_resnet, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1).squeeze()
                features_std = global_std_pool2d(x).squeeze()
                a = torch.cat((features_mean, features_std),0).squeeze()
                print(a.shape, features_mean.shape, features_std.shape)
                return a


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


class Resnet():

    def __init__(self, args: str) -> None:
        self.config: Any = get_config(args)

        size = self.config["size"]

        if size == 18:
            self.init_18()
        if size == 34:
            self.init_34()
        if size == 50:
            self.init_50()
        if size == 101:
            self.init_101()
        if size == 152:
            self.init_152()

        self.resnet = _resnet(self.resnet)

    def to(self, device: torch.device) -> None:
        self.resnet.to(device)
        self.preprocess.to(device)

    def forward(self, data: torch.Tensor)-> Any:
        with torch.no_grad():
            x = self.preprocess(data)
            return self.resnet(x)


    def init_50(self):
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()


    def init_18(self):
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()


    def init_101(self):
        self.resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet101_Weights.IMAGENET1K_V1.transforms()


    def init_34(self):
        self.resnet = torchvision.models.resnet34(weigths=torchvision.models.ResNet34_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet34_Weights.IMAGENET1K_V1.transforms()



    def init_152(self):
        self.resnet = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()


def get_resnet(args: str) -> Resnet:
    return Resnet(args)
