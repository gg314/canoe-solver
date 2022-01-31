""" Script to play two agents (including Humans) against each other """

import argparse
import numpy as np
import canoebot.agent as agent
import canoebot.utils as utils
import canoebot.encoders as encoders
from canoebot.board import GameState, Player
from scipy.stats import binom


def main():
    """Facilitate canoe games between bots and/or humans"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="show model heatmaps", action="store_true"
    )
    parser.add_argument("-n", "--num-games", type=int, default=100)
    args = parser.parse_args()
    trials = args.num_games

    encoder = encoders.ExperimentalEncoder()

    agents = {
        "random_agent1": agent.RandomAgent(),
        "random_agent2": agent.RandomAgent(),
        "neighbor_agent1": agent.NeighborAgent(),
        "neighbor_agent2": agent.NeighborAgent(),
        "greedy_agent1": agent.GreedyAgent(),
        "greedy_agent2": agent.GreedyAgent(),
        # q_agent1: agent.QAgent(utils.load_model("qtrained1"), encoders.OnePlaneEncoder())
        "ac_agent1": agent.ACAgent(utils.load_model("ac-v29"), encoder),
        "ac_agent2": agent.ACAgent(utils.load_model("ac-v1"), encoder),
        "human1": agent.Human(),
        "human2": agent.Human(),
    }

    wins = {"bot1": 0, "bot2": 0, "ties": 0}
    bots = {Player.red: agents["random_agent1"], Player.yellow: agents["ac_agent1"]}

    heatmap = np.zeros((6, 13))
    move_count = 0

    for trial in range(trials):
        if trial % 2:
            first_player = Player.red
        else:
            first_player = Player.yellow
        game = GameState.new_game(first_player)

        while not game.is_over():
            move_count += 1
            if bots[game.current_player].__class__.__name__ == "Human":
                game.print_board()
            bot_move = bots[game.current_player].select_move(game, verbose=args.verbose)
            game = game.apply_move(bot_move)
            # board_tensor = encoder.encode(game) # testing board encoder
            # time.sleep(1)
        game.print_board()
        game.current_player = Player.red
        board_tensor = encoder.encode(game)
        heatmap += board_tensor[1]  #  + board_tensor[2]

        print(f"Game {trial + 1} winner: {game.winner}")
        if game.winner == Player.red:
            wins["bot1"] += 1
        elif game.winner == Player.yellow:
            wins["bot2"] += 1
        else:
            wins["ties"] += 1

    print(game.winning_canoes)
    game.print_board()
    print(heatmap / trials)
    print(f"Average number of moves: {move_count / trials}")

    k = int(wins["bot1"])
    wins["bot1"] = wins["bot1"] / trials
    wins["bot2"] = wins["bot2"] / trials
    wins["ties"] = wins["ties"] / trials
    print(f"\r\nBrenchmark: {wins}")
    p_val = np.sum([binom.pmf(k, trials, 0.5) for k in range(k, trials + 1)])
    print(f"Probability of winning {k}/{trials} if p=0.5: {p_val:.3}")


if __name__ == "__main__":
    main()
