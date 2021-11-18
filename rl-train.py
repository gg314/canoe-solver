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

class ExperienceBuffer:
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)

class ExperienceCollector(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_actions = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []

    def record_decision(self, state, action):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_actions = []

    def to_buffer(self):
        return ExperienceBuffer(states = np.array(self.states), actions = np.array(self.actions), rewards = np.array(self.rewards))

def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    return ExperienceBuffer(combined_states, combined_actions, combined_rewards)

def load_experience(h5file):
    return ExperienceBuffer(
        states = np.array(h5file['experience']['states']),
        actions = np.array(h5file['experience']['actions']),
        rewards = np.array(h5file['experience']['rewards']))

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
    return game

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-in', required=True)
    parser.add_argument('--model-out', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--clipnorm', type=float, default=1.0)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--benchmark-trials', type=int, default=100)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    model_in_filename = args.model_in
    experience_files = args.experience
    model_out_filename = args.model_out
    benchmark_trials = args.benchmark_trials
    learning_rate = args.learning_rate
    clipnorm = args.clipnorm
    batch_size = args.bs

    agent1 = agent.PolicyAgent(utils.load_model(model_in_filename), encoders.OnePlaneEncoder())
    for exp_filename in experience_files:
        exp_buffer = load_experience(h5py.File(exp_filename))
        agent1.train(exp_buffer, learning_rate=learning_rate, clipnorm=clipnorm, batch_size=batch_size)

    utils.save_model(agent1.model, model_out_filename)

    wins = { "RL": 0, "pre": 0, "ties": 0 }
    agent2 = agent.PolicyAgent(utils.load_model(model_in_filename), encoders.OnePlaneEncoder())
    bots = {
        Player.red: agent1,
        Player.yellow: agent2,
    }

    for trial in range(benchmark_trials):
        sys.stdout.write(f"Benchmark game {trial+1}/{benchmark_trials}\r")
        sys.stdout.flush()
        game = GameState.new_game()
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        if game.winner == Player.red:
            wins['RL'] += 1
        elif game.winner == Player.yellow:
            wins['pre'] += 1
        else:
            wins['ties'] += 1
    print(f"\r\nBrenchmark: {wins}")


if __name__ == '__main__':
    main()