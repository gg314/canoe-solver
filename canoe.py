""" Script to play two agents (including Humans) against each other """

import logging
import re
from pathlib import Path

import canoebot.agent as agent
import canoebot.encoders as encoders
import canoebot.utils as utils
import click
import numpy as np
from canoebot.board import GameState, Player
from scipy.stats import binom


@click.command()
@click.option(
    "-n",
    "--num-games",
    type=click.INT,
    default=100,
    show_default=True,
    help="Number of games to play",
)
@click.option(
    "-m",
    "--mute",
    type=click.BOOL,
    default=False,
    help="Supress messages",
)
@click.option(
    "-v",
    "--verbose",
    type=click.BOOL,
    default=False,
    help="Show model heatmaps",
)
def _main(num_games: int, mute: bool, verbose: bool):
    main(num_games, mute, verbose)


def main(num_games, mute, verbose):
    """Facilitate canoe games between bots and/or humans"""
    logging.basicConfig(level=logging.INFO)

    encoder = encoders.ExperimentalEncoder()
    logging.info(f"Starting: {num_games} games{', muted' if mute else ''}")

    last_agent_id = None
    for ac in Path("./models/").glob("ac*.pt"):
        if m := re.search(r"(\d+)", str(ac)):
            if last_agent_id is None or m.group(1) > last_agent_id:
                last_agent_id = m.group(1)
                last_agent_file = ac
    print(f"Last AC agent: {last_agent_file}")

    agents = {
        "random_agent1": agent.RandomAgent(),
        "random_agent2": agent.RandomAgent(),
        "neighbor_agent1": agent.NeighborAgent(),
        "neighbor_agent2": agent.NeighborAgent(),
        "greedy_agent1": agent.GreedyAgent(),
        "greedy_agent2": agent.GreedyAgent(),
        "ac_agent1": agent.ACAgent(utils.load_model("./models/ac3.pt"), encoder),
        "ac_agent2": agent.ACAgent(utils.load_model("./models/ac5.pt"), encoder),
        "ac_agent3": agent.ACAgent(utils.load_model("./models/ac2.pt"), encoder),
        "ac_agent": agent.ACAgent(utils.load_model(last_agent_file), encoder),
        "human1": agent.Human(),
        "human2": agent.Human(),
    }

    # all_bots = [f"ac_agent{i}" for i in range(1, 2)]
    players = ["ac_agent2", "greedy_agent1"]
    # players = ["random_agent1", "neighbor_agent2"]
    agent1 = agents[players[0]]
    agent2 = agents[players[1]]

    wins = {"agent1": 0, "agent2": 0, "ties": 0}

    heatmap = np.zeros((6, 13))
    move_count = 0

    for trial in range(num_games):
        if trial % 2 == 1:
            bots = {
                Player.red: agent1,
                Player.yellow: agent2,
            }
            bots_r = {"agent1": Player.red, "agent2": Player.yellow}
        else:
            bots = {
                Player.red: agent2,
                Player.yellow: agent1,
            }
            bots_r = {"agent2": Player.red, "agent1": Player.yellow}
        game = GameState.new_game()

        while not game.is_over():
            move_count += 1
            if bots[game.current_player].__class__.__name__ == "Human":
                game.print_board()
            bot_move = bots[game.current_player].select_move(game, verbose=verbose)
            game = game.apply_move(bot_move)
            # board_tensor = encoder.encode(game) # testing board encoder
            # time.sleep(0.1)
        if not mute:
            game.print_board()
            pass
        # game.current_player = Player.red
        # board_tensor = encoder.encode(game)
        # heatmap += board_tensor[1]  #  + board_tensor[2]

        if not mute:
            print(f"Game {trial + 1} winner: {game.winner}")
        if game.winner == bots_r["agent1"]:
            wins["agent1"] += 1
        elif game.winner == bots_r["agent2"]:
            wins["agent2"] += 1
        else:
            wins["ties"] += 1

    if not mute:
        print(game.winning_canoes)
        game.print_board()
        print(heatmap / num_games)
        print(f"Average number of moves: {move_count / num_games}")

    k = int(wins["agent1"])
    wins["agent1"] = wins["agent1"] / num_games
    wins["agent2"] = wins["agent2"] / num_games
    wins["ties"] = wins["ties"] / num_games
    print(f"\r\nBrenchmark: {wins}")
    p_val = np.sum([binom.pmf(k, num_games, 0.5) for k in range(k, num_games + 1)])
    print(f"Probability of winning {k}/{num_games} if p=0.5: {p_val:.3}")


if __name__ == "__main__":
    _main()
