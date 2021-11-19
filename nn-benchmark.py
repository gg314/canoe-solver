# Benchmark model trained in nn-train.py
# Based on Deep Learning and the Game of Go, Chapter 6 (2018, Manning)

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
import numpy as np
import sys
import time
import agent
from board import GameState, Player
import encoders


def main():
    model = keras.models.load_model("./generated_models/greedy_trained_model.h5")
    encoder = encoders.OnePlaneEncoder()
    bots = {
        Player.red: agent.DeepLearningAgent(model, encoder),
        Player.yellow: agent.GreedyGreedyAgent(),
    }
    wins = { "bot1": 0, "bot2": 0, "ties": 0 }

    for trial in range(100):
        game = GameState.new_game()

        while not game.is_over():
           #  time.sleep(1)
            if bots[game.next_player] == None:
                print("HUMAN MOVE")
                raise NotImplementedError()
            else:
                bot_move = bots[game.next_player].select_move(game)
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
        
    print(game.winning_canoes)
    game.print_board()
    print(wins)

if __name__ == '__main__':
    main()

