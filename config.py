import json
from typing import Any

def get_config(path:str)-> Any:
    with open(path, "r") as f:
        return json.load(f)

