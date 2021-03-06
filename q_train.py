import argparse
import h5py
import sys

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Flatten, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import agent
import encoders
import utils
import numpy as np
from board import Player, GameState
from experience import ExperienceBuffer, ExperienceCollector, combine_experience, load_experience
import time


def simulate_game(red_player, yellow_player):
    moves = []
    game = GameState.new_game()
    agents = {
        Player.red: red_player,
        Player.yellow: yellow_player,
    }

    while not game.is_over():
        next_move = agents[game.current_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
    game.print_board()
    print(game.winner)
    print(game.winning_canoes)
    return game

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-in', required=True)
    parser.add_argument('--model-out', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--benchmark-trials', type=int, default=100)
    parser.add_argument('--temperature', '-t', type=float, default=0.35)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    model_in_filename = args.model_in
    experience_files = args.experience
    model_out_filename = args.model_out
    benchmark_trials = args.benchmark_trials
    learning_rate = args.learning_rate
    batch_size = args.bs
    temperature = args.temperature

    agent1 = agent.QAgent(utils.load_model(model_in_filename), encoders.OnePlaneEncoder())
    agent1.set_temperature(temperature)
    for exp_filename in experience_files:
        exp_buffer = load_experience(h5py.File("./generated_experience/" + exp_filename + ".h5"))
        agent1.train(exp_buffer, learning_rate=learning_rate, batch_size=batch_size)

    utils.save_model(agent1.model, model_out_filename)

    wins = { "RL": 0, "pre": 0, "ties": 0 }
    agent2 = agent.QAgent(utils.load_model("qinit"), encoders.OnePlaneEncoder())
    agent2.set_temperature(temperature)
    bots = {
        Player.red: agent1,
        Player.yellow: agent2,
    }

    for trial in range(benchmark_trials):
        sys.stdout.write(f"Benchmark game {trial+1}/{benchmark_trials}\r")
        sys.stdout.flush()
        if trial < int(benchmark_trials / 2):
            first_player = Player.red
        else:
            first_player = Player.yellow
        game = GameState.new_game(first_player)
        while not game.is_over():
            bot_move = bots[game.current_player].select_move(game)
            game = game.apply_move(bot_move)
        if game.winner == Player.red:
            wins['RL'] += 1
        elif game.winner == Player.yellow:
            wins['pre'] += 1
        else:
            wins['ties'] += 1
        game.print_board()
        print(game.winner)
        print(game.winning_canoes)
    print(f"\r\nBrenchmark: {wins}")


if __name__ == '__main__':
    main()