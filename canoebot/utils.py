""" Utilities: reduced down to saving and loading keras models

Removed: set GPU memory target"""
import tensorflow


def save_model(model, f):
    """Save agent model to file"""
    tensorflow.keras.models.save_model(model, "./generated_models/" + f + ".h5")


def load_model(f):
    """Load agent model from file"""
    return tensorflow.keras.models.load_model("./generated_models/" + f + ".h5")
