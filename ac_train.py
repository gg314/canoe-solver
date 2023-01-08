"""
ac-train.py: use self-play experience to train a new iteration of the neural
  net; benchmark against previous iteration.
"""
import re
from pathlib import Path

import click
import h5py
import numpy as np
from canoebot import agent, encoders, utils
from canoebot.board import GameState, Player, format_colored
from canoebot.experience import combine_experience, load_experience
from scipy.stats import binom


def make_progress_bar(wins, trials, total, winner):
    """Create a nice wide progress bar for fun"""
    total_bars = 100
    percent = int(total_bars * (wins / (trials)))
    pre = "=" * percent
    post = "-" * (total_bars - percent)
    sq = format_colored(winner, "■")
    print("")
    print(
        f"{trials:3}/{total:3} {sq} |{pre}|{post}| ({(100 * wins / trials):.1f}%)",
        end="\r",
    )


@click.command()
@click.option(
    "-i",
    "--model-in",
    type=click.Path(dir_okay=False, writable=True),
    help="Input model file",
)
@click.option(
    "-o",
    "--model-out",
    type=click.Path(dir_okay=False, writable=True),
    help="Output model file",
)
@click.option(
    "-e",
    "--experience-file",
    type=click.Path(dir_okay=False, writable=True),
    help="Input experience file",
)
@click.option(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.00001,
    help="Learning rate",
)
@click.option(
    "-bs",
    "--batch-size",
    type=int,
    default=128,
    help="Batch size",
)
@click.option(
    "-t",
    "--trials",
    type=int,
    default=500,
    help="Number of games to test results",
)
@click.option(
    "-m",
    "--mute",
    type=click.BOOL,
    default=False,
    help="Supress messages",
)
@click.option(
    "-g",
    "--generator",
    type=click.BOOL,
    default=False,
    help="Use generator instead of loading into memory",
)
def _main(
    model_in: Path,
    model_out: Path,
    experience_file: Path,
    trials: int,
    learning_rate: float,
    batch_size: int,
    generator: bool,
    mute: bool,
):
    if model_out is None and (m := re.search(r"(\d+)", model_in)):
        model_out = re.sub(r"(\d+)", lambda e: str(int(e.group(0)) + 1), model_in)
    if experience_file is None and (m := re.search(r"(\d+)", model_in)):
        experience_files = list(Path("./experience/").glob(f"exp{m.group(1)}*.h5"))
    else:
        experience_files = [experience_file]

    main(
        Path(model_in),
        Path(model_out),
        experience_files,
        trials,
        learning_rate,
        batch_size,
        generator,
        mute,
    )


def main(
    model_in: Path,
    model_out: Path,
    experience_files: list[Path],
    trials: int,
    learning_rate: float,
    batch_size: int,
    generator: bool,
    mute: bool,
):
    """Train CNN on experience, benchmark vs. previous iteration"""
    # model_out_filename = "ac-v" + str(int(args.model_in) + 1)  # args.model_out
    agent1 = agent.ACAgent(utils.load_model(model_in), encoders.ExperimentalEncoder())

    if False:  # generator:
        # dataset = tf.data.Dataset.from_generator(
        #     DataGenerator(experience_files),
        #     args=(batch_size,),
        #     output_signature=(
        #         tf.TensorSpec(shape=(None, 8, 6, 13), dtype=tf.int64),
        #         (
        #             tf.TensorSpec(shape=(None, 78), dtype=tf.float64),
        #             tf.TensorSpec(shape=(None,), dtype=tf.int64),
        #         ),
        #     ),
        # )

        # agent1.generator_train(dataset, learning_rate=learning_rate)
        pass
    else:
        # input("Run without generator?")
        buffers = []
        for exp_filename in experience_files:
            with h5py.File(exp_filename) as exp_file:
                print(f"Loading experience from `{exp_filename}`")
                exp_buffer = load_experience(exp_file)
                buffers.append(exp_buffer)
        all_experience = combine_experience(buffers)
        agent1.train(all_experience, learning_rate=learning_rate, batch_size=batch_size)

    utils.save_model(agent1.model, model_out)

    wins = {"new": 0, "old": 0, "ties": 0}
    agent2 = agent.ACAgent(utils.load_model(model_in), encoders.ExperimentalEncoder())

    total_moves = 0
    for trial in range(trials):
        if trial % 2:
            bots = {
                Player.red: agent1,
                Player.yellow: agent2,
            }
            bots_r = {"new": Player.red, "old": Player.yellow}
        else:
            bots = {
                Player.yellow: agent1,
                Player.red: agent2,
            }
            bots_r = {"new": Player.yellow, "old": Player.red}
        game = GameState.new_game()
        while not game.is_over():
            bot_move = bots[game.current_player].select_move(game)
            game = game.apply_move(bot_move)
            total_moves += 1
        if game.winner == bots_r["new"]:
            wins["new"] += 1
        elif game.winner == bots_r["old"]:
            wins["old"] += 1
        else:
            wins["ties"] += 1
        if not mute:
            make_progress_bar(wins["new"], trial + 1, trials, game.winner)

    k = int(wins["new"])
    k2 = int(wins["old"])
    k3 = k + k2
    wins["new"] = wins["new"] / trials
    wins["old"] = wins["old"] / trials
    wins["ties"] = wins["ties"] / trials
    print(f"\r\nBrenchmark: {wins}")
    p_val = np.sum([binom.pmf(k, k3, 0.5) for k in range(k, k3 + 1)])
    print(f"p = {p_val:.3}, average moves = {total_moves / (2 * trials)}")
    print(f"total = {k + k2}")


if __name__ == "__main__":
    _main()
