import numpy as np
from board import Move, Point

class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()

class RandomStrategy(Agent):
    def __init__(self):
        pass

    def select_move(self, game):
        previous_move = game.last_move
        open_spaces = game.board.return_open_spaces()
        return Move.play(open_spaces[np.random.choice(len(open_spaces))])

class GreedyStrategy(Agent):
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

class GreedyGreedyStrategy(Agent):
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

class IndexError(Exception):
    pass