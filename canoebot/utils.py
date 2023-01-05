""" Utilities: reduced down to saving and loading keras models

Removed: set GPU memory target"""
import logging
from pathlib import Path

import torch


def save_model(model, f: Path):
    """Save agent model to file"""
    # tensorflow.keras.models.save_model(model, "./generated_models/" + f + ".h5")
    logging.info(f"Saving model to `{f}`")
    torch.save(model, f)


def load_model(f: Path):
    """Load agent model from file"""
    # return tensorflow.keras.models.load_model("./generated_models/" + f + ".h5")
    return torch.load(f)
