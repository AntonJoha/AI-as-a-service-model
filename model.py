from torch import nn
from argparse import Namespace
from config import get_config
from typing import Any
import torch

class Model(nn.Module):


    def __init__(self, args: Namespace):
        super().__init__()
        config: Any = get_config(args.model)

        self.make_network(config)
        
        # Hidden state and cell state
        self.hn: None | torch.Tensor = None
        self.cn: None | torch.Tensor = None

    def make_network(self, config: Any) -> None:
        
        self.first: nn.Linear = nn.Linear(config["input"],config["input"])
        self.rnn: nn.LSTM = nn.LSTM(config["input"], config["hidden_size"], config["layers"], batch_first=True)
        self.last = nn.Linear = nn.Linear(config["hidden_size"],config["output"])
        self.relu = nn.ReLU()



    def forward(self,data: torch.Tensor) -> Any:
        x = self.relu(self.first(data))
        x, (self.hn,self.cn) = self.rnn(x)
        x = self.relu(x)
        return self.last(x)


def get_model(args: Namespace)-> Model:
    m = Model(args)
    return m
