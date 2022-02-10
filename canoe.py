""" Script to play two agents (including Humans) against each other """

import time
import argparse
import numpy as np
import canoebot.agent as agent
import canoebot.utils as utils
import canoebot.encoders as encoders
from canoebot.board import GameState, Player
from scipy.stats import binom

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def main():
    """Facilitate canoe games between bots and/or humans"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="show model heatmaps", action="store_true"
    )
    parser.add_argument("-m", "--mute", help="avoid messages", action="store_true")
    parser.add_argument("-n", "--num-games", type=int, default=100)
    args = parser.parse_args()
    trials = args.num_games

    encoder = encoders.RelativeEncoder()

    agents = {
        "random_agent1": agent.RandomAgent(),
        "random_agent2": agent.RandomAgent(),
        "neighbor_agent1": agent.NeighborAgent(),
        "neighbor_agent2": agent.NeighborAgent(),
        "greedy_agent1": agent.GreedyAgent(),
        "greedy_agent2": agent.GreedyAgent(),
        # q_agent1: agent.QAgent(utils.load_model("qtrained1"), encoders.OnePlaneEncoder())
        "ac_agent1": agent.ACAgent(utils.load_model("ac-v1"), encoder),
        "ac_agent2": agent.ACAgent(utils.load_model("ac-v2"), encoder),
        "ac_agent3": agent.ACAgent(utils.load_model("ac-v3"), encoder),
        "ac_agent4": agent.ACAgent(utils.load_model("ac-v4"), encoder),
        "ac_agent5": agent.ACAgent(utils.load_model("ac-v5"), encoder),
        "ac_agent6": agent.ACAgent(utils.load_model("ac-v6"), encoder),
        "ac_agent7": agent.ACAgent(utils.load_model("ac-v7"), encoder),
        "ac_agent8": agent.ACAgent(utils.load_model("ac-v8"), encoder),
        "ac_agent9": agent.ACAgent(utils.load_model("ac-v9"), encoder),
        "ac_agent10": agent.ACAgent(utils.load_model("ac-v10"), encoder),
        "ac_agent11": agent.ACAgent(utils.load_model("ac-v11"), encoder),
        "ac_agent12": agent.ACAgent(utils.load_model("ac-v12"), encoder),
        "ac_agent13": agent.ACAgent(utils.load_model("ac-v13"), encoder),
        "ac_agent14": agent.ACAgent(utils.load_model("ac-v14"), encoder),
        "ac_agent15": agent.ACAgent(utils.load_model("ac-v15"), encoder),
        "ac_agent16": agent.ACAgent(utils.load_model("ac-v16"), encoder),
        "ac_agent17": agent.ACAgent(utils.load_model("ac-v17"), encoder),
        "ac_agent18": agent.ACAgent(utils.load_model("ac-v18"), encoder),
        "ac_agent19": agent.ACAgent(utils.load_model("ac-v19"), encoder),
        "ac_agent20": agent.ACAgent(utils.load_model("ac-v20"), encoder),
        "human1": agent.Human(),
        "human2": agent.Human(),
    }

    all_bots = [f"ac_agent{i}" for i in range(1, 2)]

    win_matrix = np.full((len(all_bots), len(all_bots)), np.nan)
    for bot1idx in range(0, len(all_bots)):
        for bot2idx in range(bot1idx + 1, len(all_bots)):

            wins = {"bot1": 0, "bot2": 0, "ties": 0}

            heatmap = np.zeros((6, 13))
            move_count = 0

            for trial in range(trials):
                if trial % 2 == 1:
                    bots = {
                        Player.red: agents[all_bots[bot1idx]],
                        Player.yellow: agents[all_bots[bot2idx]],
                    }
                    bots_r = {"bot1": Player.red, "bot2": Player.yellow}
                else:
                    bots = {
                        Player.red: agents[all_bots[bot2idx]],
                        Player.yellow: agents[all_bots[bot1idx]],
                    }
                    bots_r = {"bot2": Player.red, "bot1": Player.yellow}
                game = GameState.new_game()

                while not game.is_over():
                    move_count += 1
                    if bots[game.current_player].__class__.__name__ == "Human":
                        game.print_board()
                    bot_move = bots[game.current_player].select_move(
                        game, verbose=args.verbose
                    )
                    game = game.apply_move(bot_move)
                    # board_tensor = encoder.encode(game) # testing board encoder
                    # time.sleep(0.1)
                if not args.mute:
                    game.print_board()
                    pass
                # game.current_player = Player.red
                # board_tensor = encoder.encode(game)
                # heatmap += board_tensor[1]  #  + board_tensor[2]

                if not args.mute:
                    print(f"Game {trial + 1} winner: {game.winner}")
                if game.winner == bots_r["bot1"]:
                    wins["bot1"] += 1
                elif game.winner == bots_r["bot2"]:
                    wins["bot2"] += 1
                else:
                    wins["ties"] += 1

            if not args.mute:
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
            win_matrix[bot1idx][bot2idx] = wins["bot1"]
            win_matrix[bot2idx][bot1idx] = wins["bot2"]

    # Plot red-green win/loss map
    print(win_matrix)
    cmap = colors.LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
    fig, ax = plt.subplots(1)
    p = ax.imshow(win_matrix, cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(p, ax=ax)
    ax.set_xlabel("Agent 2")
    ax.set_xticks(range(0, len(all_bots)))
    ax.set_xticklabels(f"v{40+i}" for i in range(0, len(all_bots)))
    ax.set_ylabel("Agent 1")
    ax.set_yticks(range(0, len(all_bots)))
    ax.set_yticklabels(f"v{40+i}" for i in range(0, len(all_bots)))
    ax.set_title("Agent 1 win rate")
    plt.show()


if __name__ == "__main__":
    main()
