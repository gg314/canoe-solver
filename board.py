import copy
import enum
from collections import namedtuple

class Point(namedtuple('Point', 'row col')):
    def neightbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1)
        ]

class Move():
    # optional expansions: resigning, draws etc.
    def __init__(self, point):
        assert (point is not None)
        self.point = point
        self.is_play = (self.point is not None)
    
    @classmethod
    def play(cls, point):
        return Move(point=point)

class Player(enum.Enum):
    red = 1
    yellow = 2

    @property
    def other(self):
        return Player.red if self == Player.yellow else Player.yellow

class Board():
    def __init__(self):
        self.num_rows = 6
        self.num_cols = 13
        self.reds = []
        self.yellows = []
        self.open_spaces = 61

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

    def __deepcopy__(self, memodict={}):
        copied = Board()
        copied.reds = copy.copy(self.reds)
        copied.yellows = copy.copy(self.yellows)
        copied.open_spaces = copy.copy(self.open_spaces)
        return copied

b = Board()
solns = []
print("MAKING SOLNS")
for r in range(1, b.num_rows): # \__/
    for c in range(1, b.num_cols - 2):
        if None not in (b.is_on_grid(Point(r, c)), b.is_on_grid(Point(r, c+3)), b.is_on_grid(Point(r+1, c+1)), b.is_on_grid(Point(r+1, c+2))):
            solns.append( (Point(r, c), Point(r, c+3), Point(r+1, c+1), Point(r+1, c+2) ) )
for r in range(1, b.num_rows): # /~~\
    for c in range(1, b.num_cols - 2):
        if None not in ( b.is_on_grid(Point(r, c+1)),  b.is_on_grid(Point(r, c+2)),  b.is_on_grid(Point(r+1, c)),  b.is_on_grid(Point(r+1, c+3))):
            solns.append( ( Point(r, c+1),  Point(r, c+2),  Point(r+1, c),  Point(r+1, c+3)) )
for r in range(1, b.num_rows - 2): # (
    for c in range(1, b.num_cols):
        if None not in (b.is_on_grid(Point(r, c+1)), b.is_on_grid(Point(r+1, c)), b.is_on_grid(Point(r+2, c)), b.is_on_grid(Point(r+3, c+1))):
            solns.append( (Point(r, c+1), Point(r+1, c), Point(r+2, c), Point(r+3, c+1)) )
for r in range(1, b.num_rows - 2): # )
    for c in range(1, b.num_cols):
        if None not in (b.is_on_grid(Point(r, c)), b.is_on_grid(Point(r+1, c+1)), b.is_on_grid(Point(r+2, c+1)), b.is_on_grid(Point(r+3, c))):
            solns.append( (Point(r, c), Point(r+1, c+1), Point(r+2, c+1), Point(r+3, c)) )

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        self.winner = None
        self.winning_canoes = None
        self.solns = solns
        
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
            self.winner = 0
            return True
        if self.next_player.other == Player.red: # ie current player = red
            moves = self.board.reds
            score = self.next_player.other
        else:
            moves = self.board.yellows
            score = self.next_player.other
        canoes = []
        for s in self.solns:
            if all(elem in moves for elem in s):
                canoes.append(s)
        if len(canoes) >= 2:
            for c1 in canoes:
                for c2 in canoes:
                    if not any(cc in c2 for cc in c1):
                        self.winner = score
                        self.winning_canoes = [c1, c2]
                        return True
        return False

    @classmethod
    def new_game(cls):
        board = Board()
        return GameState(board, Player.red, None, None)

def print_error(msg):
    print(f"\033[92m{msg}\033[0m")
    
def print_colored(turn, msg):
    if turn == 0:
        print(f"{msg}")
    elif turn == 1:
        print(f"\033[91m{msg}\033[0m")
    elif turn == 2:
        print(f"\033[93m{msg}\033[0m")
