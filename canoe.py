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
    q_agent1 = agent.QAgent(utils.load_model("qtrained1"), encoders.OnePlaneEncoder())
    ac_agent1 = agent.ACAgent(utils.load_model("ac-2"), encoders.RelativeEncoder())
    ac_agent2 = agent.ACAgent(utils.load_model("ac-init"), encoders.RelativeEncoder())
    human = agent.Human()

    wins = { "bot1": 0, "bot2": 0, "ties": 0 }
    bots = {
        Player.red: ac_agent1,
        Player.yellow: human
    }

    benchmark_trials = 500
    for trial in range(benchmark_trials):
        if (trial % 2):
            first_player = Player.red
        else:
            first_player = Player.yellow 
        game = GameState.new_game(first_player)

        while not game.is_over():
            game.print_board()
            bot_move = bots[game.current_player].select_move(game)
            game = game.apply_move(bot_move)
            time.sleep(1)
        game.print_board()
        
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
