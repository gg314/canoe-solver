import numpy as np
from board import Point, Player

class Encoder:
    def name(self):
        raise NotImplementedError()
        
    def encode(self, game_state):
        raise NotImplementedError()
        
    def encode_point(self, point):
        raise NotImplementedError()
        
    def decode_point_index(self, index):
        raise NotImplementedError()
        
    def num_points(self):
        raise NotImplementedError()
        
    def shape(self):
        raise NotImplementedError()

class OnePlaneEncoder(Encoder):
    def __init__(self):
        self.board_height = 6
        self.board_width = 13
        self.num_planes = 1
    
    def name(self):
        return 'oneplane'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row = r+1, col = c+1)
                if p in game_state.board.reds:
                    if next_player == Player.red:
                        board_tensor[r, c, 0] = 1
                    else:
                        board_tensor[r, c, 0] = -1
                elif p in game_state.board.yellows:
                    if next_player == Player.yellow:
                        board_tensor[r, c, 0] = 1
                    else:
                        board_tensor[r, c, 0] = -1
        return board_tensor

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)
    
    def decode_point_index(self, index):
        r = index // self.board_width
        c = index % self.board_width
        return Point(row = r+1, col = c+1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return (self.board_height, self.board_width, self.num_planes)


# def get_encoder_by_name(name):
#     module = importlib.import_module('.' + name)
#     constructor = getattr(module, 'create') # missing create(): dlgo/encoders/simple.py
#     return constructor()