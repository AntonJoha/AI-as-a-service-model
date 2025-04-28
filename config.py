import json
from typing import Any
import pickle


def get_config(path:str)-> Any:
    with open(path, "r") as f:
        return json.load(f)


def get_pickle(path:str)-> Any:
    try:
        p = "pickle/" + path + ".pkl"
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def make_pickle(path:str, data: Any)-> None:
    p = "pickle/" + path + ".pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)
