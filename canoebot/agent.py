"""
agent.py: policy strategies for Canoe
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from canoebot import board, encoders, experience
from canoebot.board import GameState, Move, Point

rng = np.random.default_rng()


class Agent:
    """Base class for Canoe agents"""

    def __init__(self):
        pass

    def select_move(self, game, verbose=False) -> Move:
        """Determine how to choose the next move"""
        raise NotImplementedError()

    def set_collector(self, collector: None | experience.ExperienceCollector) -> None:
        """Set an experience collector"""
        self.collector = collector


class Human(Agent):
    """Logic for human move input"""

    def __init__(self):
        Agent.__init__(self)
        self.encoder_test = encoders.RelativeEncoder()

    def select_move(self, game, verbose=False) -> Move:
        """Allow input() for human to select a move"""
        open_spaces = game.board.return_open_spaces()
        while True:
            human_input = input("Select an index: ")
            try:
                human_input = int(human_input)
                r = human_input // game.board.num_cols
                c = human_input % game.board.num_cols
                pt = Point(row=r + 1, col=c + 1)
                if pt in open_spaces:
                    return Move.play(pt)
                print("Error. Try again.")
            except Exception:  # pylint: disable=broad-except
                board_tensor = self.encoder_test.encode(game)
                print(board_tensor)


class RandomAgent(Agent):
    """Plays completely randomly"""

    def __init__(self):
        Agent.__init__(self)

    def select_move(self, game, verbose=False) -> Move:
        """Select a move randomly from the open spaces"""
        open_spaces = game.board.return_open_spaces()
        return Move.play(open_spaces[np.random.choice(len(open_spaces))])


class NeighborAgent(Agent):
    """Always selects a spot next to an opponent's previous move, if possible."""

    def __init__(self):
        Agent.__init__(self)

    def select_move(self, game, verbose=False):
        """Select move in accordance with policy"""
        previous_move = game.last_move
        # print(f"Previous move: {previous_move}")
        if previous_move is None:
            open_spaces = game.board.return_open_spaces()
            return Move.play(open_spaces[np.random.choice(len(open_spaces))])

        open_spaces = []
        for r in [
            previous_move.point.row - 1,
            previous_move.point.row,
            previous_move.point.row + 1,
        ]:
            for c in [
                previous_move.point.col - 1,
                previous_move.point.col,
                previous_move.point.col + 1,
            ]:
                pt = Point(r, c)
                if game.board.is_on_grid(pt) > 0 and game.board.get(pt) is None:
                    open_spaces.append(pt)
        if len(open_spaces) > 0:
            return Move.play(open_spaces[np.random.choice(len(open_spaces))])
        # else: no open neighbors, choose randomly
        open_spaces = game.board.return_open_spaces()
        return Move.play(open_spaces[np.random.choice(len(open_spaces))])


class HeatmapAgent(Agent):
    """Plays probabilistically from heatmap, if possible."""

    def __init__(self):
        Agent.__init__(self)
        self.encoder = encoders.ExperimentalEncoder()

    def select_move(self, game, verbose=False) -> Move:
        """Select move according to a particular heatmap"""
        num_moves = 6 * 13
        move_probs = np.array(
            [
                [
                    0.000,
                    0.062,
                    0.060,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.062,
                    0.056,
                    0.000,
                ],
                [
                    0.046,
                    0.062,
                    0.172,
                    0.046,
                    0.056,
                    0.538,
                    0.062,
                    0.450,
                    0.466,
                    0.356,
                    0.444,
                    0.052,
                    0.050,
                ],
                [
                    0.048,
                    0.424,
                    0.474,
                    0.352,
                    0.510,
                    0.478,
                    0.480,
                    0.520,
                    0.468,
                    0.518,
                    0.434,
                    0.058,
                    0.070,
                ],
                [
                    0.064,
                    0.056,
                    0.442,
                    0.434,
                    0.472,
                    0.530,
                    0.498,
                    0.516,
                    0.478,
                    0.480,
                    0.406,
                    0.382,
                    0.054,
                ],
                [
                    0.000,
                    0.050,
                    0.452,
                    0.388,
                    0.514,
                    0.324,
                    0.470,
                    0.472,
                    0.416,
                    0.476,
                    0.490,
                    0.050,
                    0.000,
                ],
                [
                    0.000,
                    0.000,
                    0.000,
                    0.066,
                    0.402,
                    0.066,
                    0.062,
                    0.514,
                    0.050,
                    0.388,
                    0.000,
                    0.000,
                    0.000,
                ],
            ]
        )

        eps = 1e-4  # base: 1e-4
        move_probs = np.clip(np.flatten(move_probs), eps, None)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs
        )

        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = Move.play(point)
            if game.is_valid_move(move):
                return move

        # else: no open moves
        return ValueError


class GreedyAgent(Agent):
    """Takes winning move if available, else plays next to opponent's previous move"""

    def __init__(self, encoder: None | encoders.Encoder = None):
        Agent.__init__(self)
        self.encoder = encoder
        self.collector: None | experience.ExperienceCollector = None

    def select_move(self, game, verbose=False) -> Move:
        """Select move in accordance with policy"""
        previous_move = game.last_move
        if previous_move is None:
            open_spaces = game.board.return_open_spaces()
            move = open_spaces[np.random.choice(len(open_spaces))]
            self.record_decision(game, move)
            return Move.play(move)

        winning_move = self.find_winning_move(game)
        if winning_move:
            self.record_decision(game, winning_move)
            return Move.play(winning_move)

        open_spaces = []
        for r in [
            previous_move.point.row - 1,
            previous_move.point.row,
            previous_move.point.row + 1,
        ]:
            for c in [
                previous_move.point.col - 1,
                previous_move.point.col,
                previous_move.point.col + 1,
            ]:
                pt = Point(r, c)
                if game.board.is_on_grid(pt) and game.board.get(pt) is None:
                    open_spaces.append(pt)

        if len(open_spaces) == 0:
            open_spaces = game.board.return_open_spaces()

        open_spaces = self.remove_losing_moves(game, open_spaces)
        move = open_spaces[np.random.choice(len(open_spaces))]
        self.record_decision(game, move)
        return Move.play(move)

    def find_winning_move(self, game) -> None | Move:
        """Return the first winning move, if one exists"""
        if game.moves_made <= 2 * 7:
            return None
        for candidate in game.board.return_open_spaces():
            next_state = copy.deepcopy(game).apply_move(Move.play(candidate))
            if next_state.is_over() and (next_state.winner == game.current_player):
                return candidate
        return None

    def remove_losing_moves(self, game, open_spaces):
        """Remove any move which allows the next player to immediately win"""
        okay_moves = []
        for candidate in open_spaces:
            next_state = copy.deepcopy(game).apply_move(Move.play(candidate))
            opponent_winning_move = self.find_winning_move(next_state)
            if opponent_winning_move is None:
                okay_moves.append(candidate)
        if len(okay_moves) > 0:
            return okay_moves
        else:
            return open_spaces

    def record_decision(self, game: GameState, move: Move):
        if self.collector is not None and self.encoder is not None:
            board_tensor = self.encoder.encode(game)
            if self.collector is not None:
                self.collector.record_decision(
                    state=board_tensor,
                    action=move.to_idx(),
                    estimated_value=1,
                )


# class DeepLearningAgent(Agent):
#     """This agent attempts to use Dense (fully-connected) layers to replicate
#       a policy"""

#     def __init__(self, model, encoder):
#         Agent.__init__(self)
#         self.model = model
#         self.encoder = encoder

#     def predict(self, game_state):
#         """Use keras to create policy (move probabilities)"""
#         encoded_state = self.encoder.encode(game_state)
#         input_tensor = np.array([encoded_state])  # .reshape(1, 6, 13, 1)
#         return self.model.predict(input_tensor)[0]

#     def select_move(self, game, verbose=False):
#         """Rank moves and select one to play"""
#         num_moves = self.encoder.board_width * self.encoder.board_height
#         move_probs = self.predict(game)
#         move_probs = move_probs**3
#         eps = 1e-5
#         move_probs = np.clip(move_probs, eps, 1 - eps)
#         move_probs = move_probs / np.sum(move_probs)
#         candidates = np.arange(num_moves)
#         ranked_moves = np.random.choice(
#             candidates, num_moves, replace=False, p=move_probs
#         )
#         for point_idx in ranked_moves:
#             point = self.encoder.decode_point_index(point_idx)
#             if game.is_valid_move(Move.play(point)):
#                 return Move.play(point)
#         raise ValueError


# class PolicyAgent(Agent):
#     """This agent attempts to use Dense (fully-connected) layers to replicate
#       a policy"""

#     def __init__(self, model, encoder):
#         Agent.__init__(self)
#         self.model = model
#         self.encoder = encoder
#         self.temperature = 0.0
#         self.collector = None

#     def set_collector(self, collector):
#         """Set an experience collector"""
#         self.collector = collector

#     def select_move(self, game, verbose=False):
#         """Select a move via model.predict()"""
#         num_moves = self.encoder.board_width * self.encoder.board_height
#         board_tensor = self.encoder.encode(game)
#         X = np.array([board_tensor])

#         if np.random.random() < self.temperature:
#             move_probs = np.ones(num_moves) / num_moves
#         else:
#             move_probs = self.model.predict(X)[0]

#         # Prevent move probs from getting stuck at 0 or 1.
#         eps = 1e-5
#         move_probs = np.clip(move_probs, eps, 1 - eps)
#         move_probs = move_probs / np.sum(move_probs)
#         candidates = np.arange(num_moves)
#         ranked_moves = np.random.choice(
#             candidates, num_moves, replace=False, p=move_probs
#         )
#         for point_idx in ranked_moves:
#             point = self.encoder.decode_point_index(point_idx)
#             if game.is_valid_move(Move.play(point)):
#                 if self.collector is not None:
#                     self.collector.record_decision(
#                       state=board_tensor, action=point_idx
#                       )
#                 return Move.play(point)
#         raise ValueError

#     def train(self, experience, learning_rate, clipnorm, batch_size):
#         """Train the agent's model with keras"""
#         opt = tensorflow.keras.optimizers.SGD(
#             learning_rate=learning_rate, clipnorm=clipnorm
#         )
#         self.model.compile(loss="categorical_crossentropy", optimizer=opt)
#         target_vectors = prepare_experience_data(
#             experience, self.encoder.board_width, self.encoder.board_height
#         )
#         self.model.fit(
#             experience.states, target_vectors, batch_size=batch_size, epochs=1
#         )

#     def serialize(self, h5file):
#         """Add the name of the encoder to the model's h5 file"""
#         h5file.create_group("encoder")
#         h5file["encoder"].attrs["name"] = self.encoder.name()
#         h5file.create_group("model")
#         utils.save_model_to_hdf5_group(self.model, h5file["model"])


# def prepare_experience_data(experience, board_width, board_height):
#     """Reshape experience data for keras.fit"""
#     experience_size = experience.actions.shape[0]
#     target_vectors = np.zeros((experience_size, board_width * board_height))
#     for i in range(experience_size):
#         action = experience.actions[i]
#         reward = experience.rewards[i]
#         target_vectors[i][action] = reward
#     return target_vectors


# class QAgent(Agent):
#     """An agent that attempts to use a predicted Q-function (value) for learning"""

#     def __init__(self, model, encoder):
#         Agent.__init__(self)
#         self.model = model
#         self.encoder = encoder
#         self.collector = None
#         self.temperature = 0.0
#         self.last_move_value = 0

#     def set_temperature(self, temperature):
#         """Set the agent's temperature (probability to deviate from training)"""
#         self.temperature = temperature

#     def set_collector(self, collector):
#         """Set an experience collector"""
#         self.collector = collector

#     def select_move(self, game, verbose=False):
#         """Select a move via model.predict()"""
#         board_tensor = self.encoder.encode(game)

#         moves = []
#         board_tensors = []
#         for move in game.legal_moves():
#             if not move.is_play:
#                 continue
#             moves.append(self.encoder.encode_point(move.point))
#             board_tensors.append(board_tensor)
#         if not moves:
#             raise NotImplementedError()  # never should happen

#         num_moves = len(moves)
#         board_tensors = np.array(board_tensors)
#         move_vectors = np.zeros((num_moves, self.encoder.num_points))
#         for i, move in enumerate(moves):
#             move_vectors[i][move] = 1

#         values = self.model.predict([board_tensors, move_vectors])
#         values = values.reshape(len(moves))

#         ranked_moves = self.rank_moves_eps_greedy(values)

#         for move_idx in ranked_moves:
#             point = self.encoder.decode_point_index(moves[move_idx])
#             if self.collector is not None:
#                 self.collector.record_decision(
#                     state=board_tensor, action=moves[move_idx]
#                 )
#             return Move.play(point)
#         raise ValueError

#     def train(self, experience, learning_rate=0.1, batch_size=128):
#         """Train the agent's model with keras"""
#         opt = tensorflow.keras.optimizers.SGD(learning_rate=learning_rate)
#         self.model.compile(loss="mse", optimizer=opt)

#         n = experience.states.shape[0]
#         num_moves = self.encoder.num_points
#         y = np.zeros((n,))
#         actions = np.zeros((n, num_moves))
#         for i in range(n):
#             action = experience.actions[i]
#             reward = experience.rewards[i]
#             actions[i][action] = 1
#             y[i] = reward
#         self.model.fit(
#               [experience.states, actions], y, batch_size=batch_size, epochs=1
#               )

#     def rank_moves_eps_greedy(self, values):
#         """Return a sorted, ranked list of moves"""
#         if np.random.random() < self.temperature:
#             values = np.random.random(values.shape)
#         # This ranks the moves from worst to best.
#         ranked_moves = np.argsort(values)
#         # Return them in best-to-worst order.
#         return ranked_moves[::-1]

#     def serialize(self, h5file):
#         """Add information to the model's h5 file"""
#         raise NotImplementedError()

#     def diagnostics(self):
#         """Returning testing diagnostics"""
#         return {"value": self.last_move_value}


class ACNet(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # self.convTEST2 = nn.Sequential(
        #     nn.Conv2d(8, 128, (7, 7), padding=3), nn.SELU()
        # )
        self.conv1 = nn.Sequential(nn.Conv2d(8, 128, (5, 5), padding=2), nn.SELU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=1), nn.SELU())

        self.policy_network = nn.Sequential(
            nn.Conv2d(128, 1, (1, 1), padding=0),
            nn.Flatten(),
            nn.Softmax(dim=1),
        )  # flatten to num_points, then softmax each array in the batch

        self.value_network = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.SELU(),
            nn.Conv2d(128, 1, (1, 1), padding=0),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(encoder.num_points, 256),
            nn.SELU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )  # return estimate for winner (between -1 and 1)

    def forward(self, x) -> tuple[torch.tensor, torch.tensor]:
        # x = self.convTEST2(x)
        x = self.conv1(x)
        for _ in range(2, 9):
            x = self.conv2(x)

        policy_output = self.policy_network(x)
        value_output = self.value_network(x)

        return policy_output, value_output


class ACAgent(Agent):
    """Actor critic agent: simultaneously learn:
    actor: policy probability array
    critic: value function
    """

    def __init__(self, model: ACNet, encoder: encoders.Encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder
        self.collector: None | experience.ExperienceCollector = None
        self.last_move_value: float = 0

    def select_move(self, game: board.GameState, verbose: bool = False) -> Move:
        """Select a move via model.predict()"""
        num_moves = self.encoder.num_points
        board_tensor = self.encoder.encode(game)
        X = torch.from_numpy(board_tensor).float().unsqueeze(0)
        # X = np.array(board_tensor)
        # X = torch.from_numpy(board_tensor).float().unsqueeze(0)

        actions, values = self.model.forward(X)
        move_probs = actions[0].detach().numpy()
        estimated_value = values[0][0].item()
        self.last_move_value = estimated_value

        # for rr in range(6):
        #   for cc in range(13):
        #     print(f"{move_probs[13*rr + cc]:.3f} ", end="")
        #   print(" ")

        eps = 1e-6  # base: 1e-4
        move_probs = np.clip(move_probs, eps, None)
        move_probs = np.multiply(board_tensor[3].flatten(), move_probs)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = rng.choice(
            candidates,
            game.board.open_spaces,
            replace=False,
            p=move_probs,
            shuffle=False,
        )

        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = Move.play(point)
            if game.is_valid_move(move):

                # Plot heatmaps
                if verbose:
                    for idx in game.board.illegal_spaces:
                        move_probs[idx] = np.nan
                    heatmap = move_probs.reshape((6, 13))
                    movemap = board_tensor[1] - board_tensor[2]
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(heatmap)
                    ax[1].imshow(movemap, cmap="hot")
                    ax[0].spines["top"].set_visible(False)
                    ax[0].spines["right"].set_visible(False)
                    ax[0].spines["bottom"].set_visible(False)
                    ax[0].spines["left"].set_visible(False)
                    ax[0].set_axis_off()
                    ax[1].set_axis_off()
                    ax[0].set_title(
                        f"{game.current_player} to move\n"
                        f"Chosen moves: {ranked_moves[0:6]}\n"
                        f"Actual move: {point_idx}\nEstimated value: {estimated_value}"
                    )
                    for i in range(6):
                        for j in range(13):
                            if board_tensor[3][i][j] > 0:
                                ax[0].text(
                                    j,
                                    i,
                                    13 * i + j,
                                    ha="center",
                                    va="center",
                                    color="w",
                                )
                    fig.tight_layout()
                    plt.show()

                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value,
                    )
                return move
        raise ValueError

    def train(self, experience, learning_rate, batch_size=128):
        """Train the agent's model with PyTorch"""

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )
        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.MSELoss()

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n, 1))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            policy_target[i][action] = experience.advantages[i]
            # policy_target[i][action] = reward
            value_target[i] = reward

        dataset = TensorDataset(
            torch.from_numpy(experience.states).float(),
            torch.from_numpy(policy_target).float(),
            torch.from_numpy(value_target).float(),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1):
            for idx, (x_state_batch, y_policy_batch, y_value_batch) in enumerate(
                dataloader
            ):
                policy_pred, value_pred = self.model(x_state_batch)

                loss1 = criterion1(policy_pred, y_policy_batch)
                loss2 = criterion2(value_pred, y_value_batch)
                loss = loss1 + 4 * loss2  # consider weights

                optimizer.zero_grad()  # zero the gradient buffers
                loss.backward()
                optimizer.step()

                # Every 100 batches, print the loss for this batch
                # as well as the number of examples processed so far
                if idx % 10 == 0:
                    print(
                        f"Total loss: {loss.item():>4f} "
                        f"[{(idx + 1) * len(x_state_batch):>6d}/{n:>6d}] "
                        f"{loss1.item():>3f} | {loss2.item():>3f} "
                    )

    def serialize(self, h5file):
        """Add information to the model's h5 file"""
        raise NotImplementedError()

    def diagnostics(self):
        """Return testing diagnostics"""
        return {"value": self.last_move_value}
