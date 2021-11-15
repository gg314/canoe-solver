import sys
import numpy as np
import enum
from collections import namedtuple
import copy
import time

class Player(enum.Enum):
    red = 1
    yellow = 2

    @property
    def other(self):
        return Player.red if self == Player.yellow else Player.yellow

class Point(namedtuple('Point', 'row col')):
    def neightbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1)
        ]

class Board():
    def __init__(self):
        self.num_rows = 6
        self.num_cols = 13
        self.reds = []
        self.yellows = []
        self.solns = []
        self.open_spaces = 61
        for r in range(1, self.num_rows): # \__/
            for c in range(1, self.num_cols - 2):
                if None not in (self.is_on_grid(Point(r, c)), self.is_on_grid(Point(r, c+3)), self.is_on_grid(Point(r+1, c+1)), self.is_on_grid(Point(r+1, c+2))):
                    self.solns.append( (Point(r, c), Point(r, c+3), Point(r+1, c+1), Point(r+1, c+2) ) )
        for r in range(1, self.num_rows): # /~~\
            for c in range(1, self.num_cols - 2):
                if None not in ( self.is_on_grid(Point(r, c+1)),  self.is_on_grid(Point(r, c+2)),  self.is_on_grid(Point(r+1, c)),  self.is_on_grid(Point(r+1, c+3))):
                    self.solns.append( ( Point(r, c+1),  Point(r, c+2),  Point(r+1, c),  Point(r+1, c+3)) )
        for r in range(1, self.num_rows - 2): # (
            for c in range(1, self.num_cols):
                if None not in (self.is_on_grid(Point(r, c+1)), self.is_on_grid(Point(r+1, c)), self.is_on_grid(Point(r+2, c)), self.is_on_grid(Point(r+3, c+1))):
                    self.solns.append( (Point(r, c+1), Point(r+1, c), Point(r+2, c), Point(r+3, c+1)) )
        for r in range(1, self.num_rows - 2): # )
            for c in range(1, self.num_cols):
                if None not in (self.is_on_grid(Point(r, c)), self.is_on_grid(Point(r+1, c+1)), self.is_on_grid(Point(r+2, c+1)), self.is_on_grid(Point(r+3, c))):
                    self.solns.append( (Point(r, c), Point(r+1, c+1), Point(r+2, c+1), Point(r+3, c)) )

    def print_board(self):
        print("")
        counter = 1
        for r in range(1, self.num_rows + 1):
            for c in range(1, self.num_cols + 1):
                pt = Point(r, c)
                if not self.is_on_grid(pt):
                    print("  ", end=" ")
                elif pt in self.reds:
                    print(f"\033[91m ■\033[0m", end=" ")
                    counter += 1
                elif pt in self.yellows:
                    print(f"\033[93m ■\033[0m", end=" ")
                    counter += 1
                else:
                    print(f"{counter:02d}\033[0m", end=" ")
                    counter += 1
            print("")
        print("")

        
    def place_peg(self, player, point):
        assert self.is_on_grid(point)
        assert self.get(point) is None
        if player == Player.red:
            self.reds.append(point)
        else:
            self.yellows.append(point)
        self.last_move = point
        self.open_spaces -= 1

    def return_open_spaces(self):
        open_spaces = []
        for r in range(1, self.num_rows + 1):
            for c in range(1, self.num_cols + 1):
                pt = Point(r, c)
                if self.is_on_grid(pt) and self.get(pt) is None:
                    open_spaces.append(pt)
        return open_spaces
        
    def is_on_grid(self, point):
        if point in [Point(1, 1), Point(1, 4), Point(1, 5), Point(1, 6), Point(1, 7), Point(1, 8), Point(1, 9), Point(1, 10), Point(1, 13)]:
            return False
        if point in [Point(5, 1), Point(6, 1), Point(6, 2), Point(6, 3), Point(5, 13), Point(6, 11), Point(6, 12), Point(6, 13)]:
            return False
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols

    def get(self, point):
        if point in self.reds:
            return Player.red
        elif point in self.yellows:
            return Player.yellow
        else:
            return None

class Move():
    # optional expansions: resigning, draws etc.
    def __init__(self, point):
        assert (point is not None)
        self.point = point
        self.is_play = (self.point is not None)
    
    @classmethod
    def play(cls, point):
        return Move(point=point)

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        
    def print_board(self):
        self.board.print_board()

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_peg(self.next_player, move.point)
        return GameState(next_board, self.next_player.other, self, move)

    # return 0 for tie, 1 for red win, -1 for yellow win, None for ongoing
    def is_over(self):
        if self.board.open_spaces <= 0:
            return 0
        if self.next_player == Player.yellow: # ie current player = red
            moves = self.board.reds
            score = 1
        else:
            moves = self.board.yellows
            score = -1
        canoes = []
        for s in self.board.solns:
            if all(elem in moves for elem in s):
                canoes.append(s)
        if len(canoes) >= 2:
            for c1 in canoes:
                for c2 in canoes:
                    if not any(cc in c2 for cc in c1):
                        print(canoes)
                        return score
        return None

    @classmethod
    def new_game(cls):
        board = Board()
        return GameState(board, Player.red, None, None)


class RandomStrategy:
    def __init__(self):
        pass

    def select_move(self, game):
        previous_move = game.last_move
        open_spaces = game.board.return_open_spaces()
        return Move.play(open_spaces[np.random.choice(len(open_spaces))])

class GreedyStrategy:
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

class IndexError(Exception):
    pass

   


def print_error(msg):
    print(f"\033[92m{msg}\033[0m")
    
def print_colored(turn, msg):
    if turn == 0:
        print(f"{msg}")
    elif turn == 1:
        print(f"\033[91m{msg}\033[0m")
    elif turn == 2:
        print(f"\033[93m{msg}\033[0m")


def main():
    game = GameState.new_game()
    bots = {
        Player.red: GreedyStrategy(),
        Player.yellow: RandomStrategy()
    }
    while game.is_over() is None:
        # time.sleep(0.1)
        print(chr(27) + "[2J")
        bot_move = bots[game.next_player].select_move(game)
        game = game.apply_move(bot_move)
        game.print_board()
    print(game.is_over())

if __name__ == '__main__':
    main()
