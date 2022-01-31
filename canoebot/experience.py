"""
experience.py: record game history and results
"""

from typing import List
import h5py
import numpy as np


class DataGenerator:
    """experience data consumed by keras.train as a generator"""

    def __init__(self, filenames):
        self.filenames = filenames
        self.exp_buffer = ExperienceBuffer([], [], [], [])
        self.idx = 0

    def __call__(self, batch_size):
        keep_running = True
        while keep_running:
            if len(self.exp_buffer) - self.idx < batch_size:
                self.exp_buffer.states = self.exp_buffer.states[self.idx :]
                self.exp_buffer.actions = self.exp_buffer.actions[self.idx :]
                self.exp_buffer.rewards = self.exp_buffer.rewards[self.idx :]
                self.exp_buffer.advantages = self.exp_buffer.advantages[self.idx :]
                if len(self.filenames) == 0:
                    keep_running = False
                    batch_size = len(self.exp_buffer)
                else:
                    current_file = self.filenames.pop(0)
                    with h5py.File(
                        "./generated_experience/" + current_file + ".h5"
                    ) as exp_file:
                        if len(self.exp_buffer):
                            self.exp_buffer = combine_experience(
                                [self.exp_buffer, load_experience(exp_file)]
                            )
                        else:
                            self.exp_buffer = load_experience(exp_file)
                self.idx = 0

            num_moves = 6 * 13
            states = self.exp_buffer.states[self.idx : self.idx + batch_size]
            actions = self.exp_buffer.actions[self.idx : self.idx + batch_size]
            rewards = self.exp_buffer.rewards[self.idx : self.idx + batch_size]
            advantages = self.exp_buffer.advantages[self.idx : self.idx + batch_size]
            policy_targets = np.zeros((batch_size, num_moves))
            value_targets = np.zeros((batch_size,))
            for i in range(batch_size):
                action = actions[i]
                reward = rewards[i]
                value_targets[i] = reward
                policy_targets[i][action] = advantages[i]

            self.idx += batch_size
            # print(f"Sending: {states} pt: {policy_targets}, vt: {value_targets}")
            yield (states, (policy_targets, value_targets))

    def throw(self, typ=None, val=None, tb=None):  # pylint: disable=invalid-name
        """StopIteration when out of samples"""
        raise StopIteration


class ExperienceBuffer:
    """bundles of states, actions, rewards, advantages"""

    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions  # index of selected move
        self.rewards = rewards  # won or lost
        self.advantages = advantages

    def serialize(self, h5file):
        """Write Experience buffer to labeled h5 dataset"""
        h5file.create_group("experience")
        h5file["experience"].create_dataset("states", data=self.states, dtype="i2")
        h5file["experience"].create_dataset("actions", data=self.actions, dtype="i8")
        h5file["experience"].create_dataset("rewards", data=self.rewards, dtype="i2")
        h5file["experience"].create_dataset(
            "advantages", data=self.advantages, dtype="float32"
        )

    def __len__(self):
        return len(self.states)


class ExperienceCollector(object):
    """Combine containers for storing game history (experience)"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def begin_episode(self):
        """Initialize empty containers"""
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=0):
        """Record an action as one datapoint of experience"""
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        """Commit current episode to the list of all experience"""
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []


def combine_experience(collectors: List[ExperienceCollector]) -> ExperienceBuffer:
    """Combine multiple collectors into one"""
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])
    return ExperienceBuffer(
        combined_states, combined_actions, combined_rewards, combined_advantages
    )


def load_experience(h5file) -> ExperienceBuffer:
    """Load experience to an ExperienceBuffer from an h5 file"""
    return ExperienceBuffer(
        states=np.array(h5file["experience"]["states"]),
        actions=np.array(h5file["experience"]["actions"]),
        rewards=np.array(h5file["experience"]["rewards"]),
        advantages=np.array(h5file["experience"]["advantages"]),
    )
