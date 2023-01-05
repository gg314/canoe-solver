"""
encoders.py: encode a GameState into NN/numpy data structures
"""

import numpy as np
from canoebot.board import GameState, Player, Point


class Encoder:
    """Base class for encoding a game into feature planes"""

    def name(self) -> str:
        """Give a human-readable name to the encoder"""
        raise NotImplementedError()

    def encode(self, game_state: GameState) -> np.array:
        """Initialize the encoding"""
        raise NotImplementedError()

    def encode_point(self, point: Point) -> int:
        """Encode a specific Point to an integer"""
        raise NotImplementedError()

    def decode_point_index(self, index: int) -> Point:
        """Convert an integer back to a Point"""
        raise NotImplementedError()

    @property
    def num_points(self) -> int:
        """Return the dimensionality (total number of points) on a board"""
        raise NotImplementedError()

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return tuple for the shape of the feature planes, including the number
        of planes"""
        raise NotImplementedError()


class OnePlaneEncoder(Encoder):
    """Encode all information on one board of 0s, 1s, and -1s"""

    def __init__(self):
        self.board_height = 6
        self.board_width = 13
        self.num_planes = 1

    def name(self):
        return "oneplane"

    def encode(self, game_state: GameState) -> np.array:
        board_tensor = np.zeros(self.shape)
        current_player = game_state.current_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                if game_state.board.get(p) == current_player:
                    board_tensor[r, c, 0] = 1
                else:
                    board_tensor[r, c, 0] = -1
        return board_tensor

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        r = index // self.board_width
        c = index % self.board_width
        return Point(row=r + 1, col=c + 1)

    @property
    def num_points(self):
        return self.board_width * self.board_height

    @property
    def shape(self):
        return (self.board_height, self.board_width, self.num_planes)


class SixPlaneEncoder(Encoder):
    """Early encoder which keeps track of whether a point completes a canoe

    0. red points
    1. yellow points
    2. completes red canoe
    3. completes yellow canoe
    4. red plays next
    5. yellow plays next
    """

    def __init__(self):
        self.board_height = 6
        self.board_width = 13
        self.num_planes = 6

    def name(self):
        return "sixplane"

    def encode(self, game_state: GameState) -> np.array:
        board_tensor = np.zeros(self.shape)
        current_player = game_state.current_player
        if current_player == Player.red:
            board_tensor[:, :, 4] = 1
        else:
            board_tensor[:, :, 5] = 1

        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                occupied = game_state.board.get(p)
                if occupied is None:  # empty, check if completes canoe
                    if game_state.completes_canoe(p, Player.red):
                        board_tensor[r][c][2] = 1
                    if game_state.completes_canoe(p, Player.yellow):
                        board_tensor[r][c][3] = 1
                elif occupied == Player.red:
                    board_tensor[r][c][0] = 1
                else:
                    board_tensor[r][c][1] = 1
        return board_tensor

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        r = index // self.board_width
        c = index % self.board_width
        return Point(row=r + 1, col=c + 1)

    @property
    def num_points(self):
        return self.board_width * self.board_height

    @property
    def shape(self):
        return (self.board_height, self.board_width, self.num_planes)


class RelativeEncoder(Encoder):
    """Keep track of whether a point completes a canoe, relative to current_player

    0. current player: 0 = red, 1 = yellow
    1. current_player pegs
    2. opponent pegs
    3. open spots
    4. completes current_player canoe
    5. completes oppoonent canoe
    """

    def __init__(self):
        self.board_height = 6
        self.board_width = 13
        self.num_planes = 6

    def name(self):
        return "relative"

    def encode(self, game_state: GameState) -> np.array:
        board_tensor = np.zeros(self.shape)
        current_player = game_state.current_player
        next_player = game_state.current_player.other
        if current_player == Player.yellow:
            board_tensor[0, :, :] = 1

        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                occupied = game_state.board.get(p)
                if occupied is None:  # empty, check if completes canoe
                    if game_state.board.is_on_grid(p):
                        board_tensor[3][r][c] = 1
                    if game_state.completes_canoe(p, current_player):
                        board_tensor[4][r][c] = 1
                    if game_state.completes_canoe(p, next_player):
                        board_tensor[5][r][c] = 1
                elif occupied == current_player:
                    board_tensor[1][r][c] = 1
                else:
                    board_tensor[2][r][c] = 1
        return board_tensor.unsqueeze(0)

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        r = index // self.board_width
        c = index % self.board_width
        return Point(row=r + 1, col=c + 1)

    @property
    def num_points(self):
        return self.board_width * self.board_height

    @property
    def shape(self):
        return (self.num_planes, self.board_height, self.board_width)


class ExperimentalEncoder(Encoder):
    """Keep track of whether a point completes a canoe or two, relatively

    0. current player: 0 = red, 1 = yellow
    1. current_player pegs
    2. opponent pegs
    3. open spots
    4. completes current_player canoe
    5. completes oppoonent canoe
    6. is in current_player canoe
    7. is in opponent canoe
    """

    def __init__(self):
        self.board_height = 6
        self.board_width = 13
        self.num_planes = 8

    def name(self):
        return "eightplane"

    def encode(self, game_state: GameState) -> np.array:
        board_tensor = np.zeros(self.shape)
        current_player = game_state.current_player
        next_player = game_state.current_player.other
        if current_player == Player.yellow:
            board_tensor[0, :, :] = 1

        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                occupied = game_state.board.get(p)
                if occupied is None:  # empty, check if completes canoe
                    if game_state.board.is_on_grid(p):
                        board_tensor[3][r][c] = 1
                    if game_state.completes_canoe(p, current_player):
                        board_tensor[4][r][c] = 1
                    if game_state.completes_canoe(p, next_player):
                        board_tensor[5][r][c] = 1
                elif occupied == current_player:
                    board_tensor[1][r][c] = 1
                    if game_state.is_in_canoe(p, current_player):
                        board_tensor[6][r][c] = 1
                else:
                    board_tensor[2][r][c] = 1
                    if game_state.is_in_canoe(p, next_player):
                        board_tensor[7][r][c] = 1
        return board_tensor

    def encode_point(self, point) -> int:
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index) -> Point:
        r = index // self.board_width
        c = index % self.board_width
        return Point(row=r + 1, col=c + 1)

    @property
    def num_points(self) -> int:
        return self.board_width * self.board_height

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_planes, self.board_height, self.board_width)


# def get_encoder_by_name(name):
#   module = importlib.import_module('.' + name)
#   constructor = getattr(module, 'create') # missing create(): dlgo/encoders/simple.py
#   return constructor()
