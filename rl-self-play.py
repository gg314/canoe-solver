import argparse

import h5py

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


def simulate_game(red_player, yellow_player):
    moves = []
    game = GameState.new_game()
    agents = {
        Player.red: red_player,
        Player.yellow: yellow_player,
    }

    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
    game.print_board()
    print(game.winner)
    print(game.winning_canoes)
    return game

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-in', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--experience-out', required=True)
    args = parser.parse_args()
    model_in_filename = args.model_in
    experience_filename = args.experience_out
    num_games = args.num_games

    agent1 = agent.PolicyAgent(utils.load_model(model_in_filename), encoders.OnePlaneEncoder())
    agent2 = agent.PolicyAgent(utils.load_model(model_in_filename), encoders.OnePlaneEncoder())
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for i in range(args.num_games):
        print(f"Simulating game {i + 1}/{num_games}...")
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)
        if game_record.winner == Player.red:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        elif game_record.winner == Player.yellow:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        else:
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)

    experience = combine_experience([collector1, collector2])
    with h5py.File("./generated_experience/" + experience_filename + ".h5", 'w') as experience_outf:
        experience.serialize(experience_outf)


if __name__ == '__main__':
    main()