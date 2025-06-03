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
        self.first: nn.Linear = nn.Linear(config["input"],int(config["input"]/2))
        self.query: nn.Linear = nn.Linear(config["input"], int(config["input"]/2))
        self.value: nn.Linear = nn.Linear(config["input"], int(config["input"]/2))

        self.rnn: nn.LSTM = nn.LSTM(int(config["input"]/10), config["hidden_size"], config["layers"], batch_first=True)
        self.attention = nn.MultiheadAttention(int(config["input"]/2), 16, batch_first=True)
        self.last =  nn.Linear(int(config["input"]/2),config["output"])
        self.relu = nn.ReLU()
        self.hidden_size= config["hidden_size"]

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


    def forward(self,data: torch.Tensor) -> Any:
        key = self.first(data)
        query = self.query(data)
        value = self.value(data)
        

        x, _ = self.attention(query, key, value)

        x = self.relu(x)
        return self.last(x)

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()
        if torch.cuda.is_available():
            self.cuda()
            self.cpu()
            print("USING CPU FIX LATER")
        else:
            self.cpu()


def get_model(args: Namespace)-> Model:
    m = Model(args)
    return m
