import sys
import numpy as np
import time
import agent
from board import GameState, Player




def main():

    wins = { "bot1": 0, "bot2": 0, "ties": 0 }
    bots = {
        Player.red: agent.GreedyGreedyStrategy(),
        Player.yellow: agent.GreedyStrategy()
    }

    for trial in range(100):
        game = GameState.new_game()

        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
            # game.print_board()
        
        # print(chr(27) + "[2J")
        print(f"Game {trial} winner: {game.winner}")
        if game.winner == Player.red:
            wins['bot1'] += 1
        elif game.winner == Player.yellow:
            wins['bot2'] += 1
        else:
            wins['ties'] += 1
        
    print(game.winning_canoes)
    game.print_board()
    print(wins)

if __name__ == '__main__':
    main()
