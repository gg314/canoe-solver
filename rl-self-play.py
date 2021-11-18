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
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


if __name__ == '__main__':
    main()