import numpy as np
from board import Move, Point
import sys
import utils
import encoders
from tensorflow.keras.optimizers import SGD

class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()

class RandomAgent(Agent):
    def __init__(self):
        pass

    def select_move(self, game):
        previous_move = game.last_move
        open_spaces = game.board.return_open_spaces()
        return Move.play(open_spaces[np.random.choice(len(open_spaces))])

class GreedyAgent(Agent):
    def __init__(self):
        pass

    def select_move(self, game):
        previous_move = game.last_move
        # print(f"Previous move: {previous_move}")
        if previous_move == None:
            open_spaces = game.board.return_open_spaces()
            return Move.play(open_spaces[np.random.choice(len(open_spaces))])
        
        open_spaces = []
        for r in [previous_move.point.row - 1, previous_move.point.row, previous_move.point.row + 1]:
            for c in [previous_move.point.col - 1, previous_move.point.col, previous_move.point.col + 1]:
                pt = Point(r, c)
                try:
                    if game.board.is_on_grid(pt) > 0 and game.board.get(pt) is None:
                        open_spaces.append(pt)
                except: pass
        if len(open_spaces) > 0:
            return Move.play(open_spaces[np.random.choice(len(open_spaces))])
        else: # no open neighbors, choose randomly
            open_spaces = game.board.return_open_spaces()
            return Move.play(open_spaces[np.random.choice(len(open_spaces))])

class GreedyGreedyAgent(Agent):
    def __init__(self):
        pass

    def select_move(self, game):
        previous_move = game.last_move
        # print(f"Previous move: {previous_move}")
        if previous_move == None:
            open_spaces = game.board.return_open_spaces()
            return Move.play(open_spaces[np.random.choice(len(open_spaces))])
        
        winning_move = self.find_winning_move(game)
        if winning_move:
            return Move.play(winning_move)

        open_spaces = []
        for r in [previous_move.point.row - 1, previous_move.point.row, previous_move.point.row + 1]:
            for c in [previous_move.point.col - 1, previous_move.point.col, previous_move.point.col + 1]:
                pt = Point(r, c)
                try:
                    if game.board.is_on_grid(pt) > 0 and game.board.get(pt) is None:
                        open_spaces.append(pt)
                except: pass
        if len(open_spaces) == 0:
            open_spaces = game.board.return_open_spaces()
        
        open_spaces = self.remove_losing_moves(game, open_spaces)
        return Move.play(open_spaces[np.random.choice(len(open_spaces))])

    def find_winning_move(self, game):
        for candidate in game.board.return_open_spaces():
            next_state = game.apply_move(Move.play(candidate))
            if next_state.is_over() and next_state.winner == next_state.next_player.other:
                return candidate
        return None
    
    def remove_losing_moves(self, game, open_spaces):
        okay_moves = []
        for candidate in open_spaces:
            next_state = game.apply_move(Move.play(candidate))
            opponent_winning_move = self.find_winning_move(next_state)
            if opponent_winning_move == None:
                okay_moves.append(candidate)
        if len(okay_moves) > 0:
            return okay_moves
        else:
            return open_spaces

class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state]) # .reshape(1, 6, 13, 1)
        return self.model.predict(input_tensor)[0]

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)
        move_probs = move_probs ** 3
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1-eps)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(Move.play(point)):
                return Move.play(point)

class PolicyAgent(Agent):
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.temperature = 0.0
        self.collector = None

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        board_tensor = self.encoder.encode(game_state)
        X = np.array([board_tensor])

        if np.random.random() < self.temperature:
            move_probs = np.ones(num_moves) / num_moves
        else:
            move_probs = self.model.predict(X)[0]

        # Prevent move probs from getting stuck at 0 or 1.
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace = False, p = move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(Move.play(point)):
                if self.collector is not None:
                    self.collector.record_decision(state=board_tensor, action=point_idx)
                return Move.play(point)

    def train(self, experience, learning_rate, clipnorm, batch_size):
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=learning_rate, clipnorm=clipnorm))
        target_vectors = prepare_experience_data(experience, self.encoder.board_width, self.encoder.board_height)
        self.model.fit(experience.states, target_vectors, batch_size=batch_size, epochs=1)
        
    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file.create_group('model')
        utils.save_model_to_hdf5_group(self.model, h5file['model'])

# make Experience suitable for keras.fit
def prepare_experience_data(experience, board_width, board_height):
    experience_size = experience.actions.shape[0]
    target_vectors = np.zeros((experience_size, board_width * board_height))
    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        target_vectors[i][action] = reward
    return target_vectors


class IndexError(Exception):
    pass










class QAgent(Agent):
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0
#        self.last_move_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)

        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            moves.append(self.encoder.encode_point(move.point))
            board_tensors.append(board_tensor)
        if not moves:
            raise NotImplementedError() # never should happen

        num_moves = len(moves)
        board_tensors = np.array(board_tensors)
        move_vectors = np.zeros( (num_moves, self.encoder.num_points()) )
        for i, move in enumerate(moves):
            move_vectors[i][move] = 1

        values = self.model.predict( [board_tensors, move_vectors] )
        values = values.reshape(len(moves))

        ranked_moves = self.rank_moves_eps_greedy(values)

        for move_idx in ranked_moves:
            point = self.encoder.decode_point_index(moves[move_idx])
            if self.collector is not None:
                self.collector.record_decision(state=board_tensor, action=moves[move_idx])
            return Move.play(point)

    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature:
            values = np.random.random(values.shape)
        # This ranks the moves from worst to best.
        ranked_moves = np.argsort(values)
        # Return them in best-to-worst order.
        return ranked_moves[::-1]

    def train(self, experience, learning_rate=0.1, batch_size=128):
        opt = SGD(learning_rate=learning_rate)
        self.model.compile(loss='mse', optimizer=opt)

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        y = np.zeros((n,))
        actions = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            actions[i][action] = 1
            y[i] = reward
        self.model.fit( [experience.states, actions], y, batch_size=batch_size, epochs=1)

    def serialize(self, h5file):
        raise NotImplementedError()

    def diagnostics(self):
        return {'value': self.last_move_value}


def load_q_agent(h5file):
    model = utils.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    # board_width = h5file['encoder'].attrs['board_width']
    # board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.OnePlaneEncoder()
    return QAgent(model, encoder)

def load_policy_agent(h5file):
    model = utils.load_model_from_hdf5_group(h5file['model'], custom_objects={})
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    # board_width = h5file['encoder'].attrs['board_width']
    # board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.OnePlaneEncoder()
    return PolicyAgent(model, encoder)