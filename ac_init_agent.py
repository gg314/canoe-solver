"""
ac-init-agent.py: initialize a keras model for a Canoe agent
"""

from pathlib import Path

import click
import torch

# from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
# from tensorflow.keras.models import Model
from canoebot import encoders, utils
from canoebot.agent import ACNet
from torchsummary import summary


@click.command()
@click.option(
    "-o",
    "--output-filename",
    type=click.Path(dir_okay=False, writable=True),
    required=False,
    default="ac1.pt",
    help="Output file for model",
)
def _main(output_filename) -> None:
    main(Path(output_filename))


def main(output_filename: Path) -> None:
    """Creates the bare CNN model. Encoder details may be irrelevant.

    data_format should be "channels_first" for GPU
    """
    encoder = encoders.ExperimentalEncoder()  # or RelativeEncoder

    print(f"Model input shape: {encoder.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACNet(encoder).to(device)
    summary(model, encoder.shape)

    output_filename = Path("./models/") / output_filename
    utils.save_model(model, output_filename)


if __name__ == "__main__":
    _main()
