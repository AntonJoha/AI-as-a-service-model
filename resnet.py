import torchvision
from argparse import Namespace
from config import get_config
from typing import Any
import torch


class Resnet():

    def __init__(self, args: Namespace) -> None:
        self.config: Any = get_config(args.resnet)

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

    def forward(self, data: torch.Tensor)-> Any:
        x = self.preprocess(data)
        return self.resnet(x)


    def init_50(self):
        self.resnet = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()


    def init_18(self):
        self.resnet = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()


    def init_101(self):
        self.resnet = torchvision.models.resnet101(torchvision.models.ResNet101_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet101_Weights.IMAGENET1K_V1.transforms()


    def init_34(self):
        self.resnet = torchvision.models.resnet34(torchvision.models.ResNet34_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet34_Weights.IMAGENET1K_V1.transforms()



    def init_152(self):
        self.resnet = torchvision.models.resnet152(torchvision.models.ResNet152_Weights.DEFAULT)
        self.preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()


def get_resnet(args: Namespace) -> Resnet:
    return Resnet(args)
