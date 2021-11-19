import argparse
import sys
import numpy as np
import time
import agent
import utils
import encoders
from board import GameState, Player


def main():


    random_agent1 = agent.RandomAgent()
    neighbor_agent1 = agent.NeighborAgent()
    greedy_agent1 = agent.GreedyAgent()
    ac_agent1 = agent.ACAgent(utils.load_model("actrained1"), encoders.OnePlaneEncoder())

    wins = { "bot1": 0, "bot2": 0, "ties": 0 }
    bots = {
        Player.red: ac_agent1,
        Player.yellow: random_agent1
    }

    benchmark_trials = 200
    for trial in range(benchmark_trials):
        if trial < int(benchmark_trials / 2):
            first_player = Player.red
        else:
            first_player = Player.yellow 
        game = GameState.new_game(first_player)

        while not game.is_over():
            bot_move = bots[game.current_player].select_move(game)
            game = game.apply_move(bot_move)
            # game.print_board()
        
        # print(chr(27) + "[2J")
        print(f"Game {trial + 1} winner: {game.winner}")
        if game.winner == Player.red:
            wins['bot1'] += 1
        elif game.winner == Player.yellow:
            wins['bot2'] += 1
        else:
            wins['ties'] += 1
    
    wins['bot1'] = wins['bot1'] / benchmark_trials
    wins['bot2'] = wins['bot2'] / benchmark_trials
    wins['ties'] = wins['ties'] / benchmark_trials
    print(game.winning_canoes)
    game.print_board()
    print(wins)

if __name__ == '__main__':
    main()
