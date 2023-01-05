"""
board.py: logic for Canoe
"""

import enum
from collections import namedtuple

import numpy as np
from typing_extensions import Self

__all__ = ["Board", "GameState", "Move"]


class Point(namedtuple("Point", "row col")):
    """Point(x, y), 1-based indexing"""

    def neighbors(self):
        """Return neighboring points (excluding diagonals)"""
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

    def to_idx(self):
        """convert Point(1, 1) to 0, etc."""
        return 13 * (self.row - 1) + (self.col - 1)


class Move:
    """Encode a potential move. Left generic in case passing were an option"""

    # optional expansions: resigning, draws etc.
    def __init__(self, point: Point):
        assert point is not None
        self.point = point
        self.is_play = self.point is not None

    @classmethod
    def play(cls, point):
        """Turns a cell into a Move()

        Future implementations could include pass(), relocate(), etc
        """
        return Move(point=point)


class Player(enum.Enum):
    """Simple enum for two-player games"""

    red = 1
    yellow = 2

    @property
    def other(self):
        """Simple helper for toggling to the other player"""
        return Player.red if self == Player.yellow else Player.yellow


class Board:
    """Contains information about the state of the board"""

    def __init__(self):
        self.num_rows = 6
        self.num_cols = 13
        self.reds = np.zeros(78, dtype=bool)
        self.yellows = np.zeros(78, dtype=bool)
        self.legal_spaces = np.ones(78, dtype=bool)
        for i in (0, 3, 4, 5, 6, 7, 8, 9, 12, 52, 64, 65, 66, 67, 75, 76, 77):
            self.legal_spaces[i] = 0
        self.open_spaces = 61
        self.last_move = None

    def print_board(self, winning_canoes=None):
        """Print the board to the command line

        Optionally: fill in the winning canoes with solid indicators"""
        print("")
        counter = 0
        for r in range(1, self.num_rows + 1):
            for c in range(1, self.num_cols + 1):
                pt = Point(r, c)
                symbol = "□"
                if winning_canoes is not None:
                    if any(counter in wc for wc in winning_canoes):
                        symbol = "■"
                if not self.is_on_grid(pt):
                    print("  ", end=" ")
                elif self.reds[counter]:
                    print(f"\033[91m {symbol}\033[0m", end=" ")
                elif self.yellows[counter]:
                    print(f"\033[93m {symbol}\033[0m", end=" ")
                else:
                    print(f"{counter:02d}\033[0m", end=" ")
                counter += 1
            print("")
        print("")

    def place_peg(self, player: Player, point: Point):
        """Fill in a cell with a peg"""
        assert self.is_on_grid(point)
        assert self.get(point) is None
        if player == Player.red:
            self.reds[point.to_idx()] = True
        else:
            self.yellows[point.to_idx()] = True
        self.last_move = point
        self.open_spaces -= 1

    def return_open_spaces(self):
        """Return all legal, open cells"""
        open_spaces = []
        for r in range(1, self.num_rows + 1):
            for c in range(1, self.num_cols + 1):
                pt = Point(r, c)
                if self.is_on_grid(pt) and self.get(pt) is None:
                    open_spaces.append(pt)
        return open_spaces

    def is_on_grid(self, point) -> bool:
        """Given a point, make sure it is in the set of legal cells

        Note: this function is board-specific to standard Canoe
        """
        idx = point.to_idx()
        try:
            if not self.legal_spaces[idx]:
                return False
            return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols
        except Exception:
            return False

    def get(self, point: Point) -> None | Player:
        """Given a point, return the player"""
        id = point.to_idx()

        if self.reds[id]:
            return Player.red
        elif self.yellows[id]:
            return Player.yellow
        else:
            return None

    def __deepcopy__(self, memodict={}):
        """Overwrite deepcopy with quicker function"""
        # pylint: disable=dangerous-default-value
        copied = Board()
        copied.reds = np.copy(self.reds)
        copied.yellows = np.copy(self.yellows)
        copied.open_spaces = np.copy(self.open_spaces)
        return copied


def make_solns():
    """Create a dict eg solns[5] for all canoes containing index 5"""
    # pylint: disable=too-many-branches

    b = Board()
    all_solutions = []
    print("Making solutions...")
    for r in range(1, b.num_rows):  # \__/
        for c in range(1, b.num_cols):
            if False not in (
                b.is_on_grid(Point(r, c)),
                b.is_on_grid(Point(r, c + 3)),
                b.is_on_grid(Point(r + 1, c + 1)),
                b.is_on_grid(Point(r + 1, c + 2)),
            ):
                all_solutions.append(
                    (
                        Point(r, c).to_idx(),
                        Point(r, c + 3).to_idx(),
                        Point(r + 1, c + 1).to_idx(),
                        Point(r + 1, c + 2).to_idx(),
                    )
                )
    for r in range(1, b.num_rows):  # /~~\
        for c in range(1, b.num_cols):
            if False not in (
                b.is_on_grid(Point(r, c + 1)),
                b.is_on_grid(Point(r, c + 2)),
                b.is_on_grid(Point(r + 1, c)),
                b.is_on_grid(Point(r + 1, c + 3)),
            ):
                all_solutions.append(
                    (
                        Point(r, c + 1).to_idx(),
                        Point(r, c + 2).to_idx(),
                        Point(r + 1, c).to_idx(),
                        Point(r + 1, c + 3).to_idx(),
                    )
                )
    for r in range(1, b.num_rows):  # (
        for c in range(1, b.num_cols):
            if False not in (
                b.is_on_grid(Point(r, c + 1)),
                b.is_on_grid(Point(r + 1, c)),
                b.is_on_grid(Point(r + 2, c)),
                b.is_on_grid(Point(r + 3, c + 1)),
            ):
                all_solutions.append(
                    (
                        Point(r, c + 1).to_idx(),
                        Point(r + 1, c).to_idx(),
                        Point(r + 2, c).to_idx(),
                        Point(r + 3, c + 1).to_idx(),
                    )
                )
    for r in range(1, b.num_rows):  # )
        for c in range(1, b.num_cols):
            if False not in (
                b.is_on_grid(Point(r, c)),
                b.is_on_grid(Point(r + 1, c + 1)),
                b.is_on_grid(Point(r + 2, c + 1)),
                b.is_on_grid(Point(r + 3, c)),
            ):
                all_solutions.append(
                    (
                        Point(r, c).to_idx(),
                        Point(r + 1, c + 1).to_idx(),
                        Point(r + 2, c + 1).to_idx(),
                        Point(r + 3, c).to_idx(),
                    )
                )
    soln_counter = 0
    solutions_dict = {}
    for idx in range(78):
        solutions_dict[idx] = []
        for soln in all_solutions:
            if idx in soln:
                tuple3 = tuple(el for el in soln if el != idx)
                solutions_dict[idx].append(tuple3)
                soln_counter += 1
    print(f"{soln_counter} solutions saved.")
    return all_solutions, solutions_dict


solns, solns_dict = make_solns()


class GameState:
    """All info about current state of the game"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, board, current_player, move):  # previous, move):
        self.board = board
        self.current_player = current_player
        self.last_move = move
        self.winner = None
        self.winning_canoes = None
        self.solns = solns
        self.solns_dict = solns_dict

    def print_board(self):
        """Helper function to print the board"""
        self.board.print_board(self.winning_canoes)

    def apply_move(self, move: Move) -> Self:
        """Apply a chosen move and return the resulting game"""
        if move.is_play:
            # next_board = copy.deepcopy(self.board)
            self.board.place_peg(self.current_player, move.point)
            self.current_player = self.current_player.other
            self.last_move = move
        # return GameState(next_board, self.current_player.other, move)
        return self

    def completes_canoe(self, pt: Point, player: Player) -> bool:
        """Return whether a Point pt completes a canoe for Player player"""
        if player == Player.red:
            moves = self.board.reds
        else:
            moves = self.board.yellows
        pt_idx = pt.to_idx()
        for s in self.solns_dict[pt_idx]:
            if all(moves[elem] for elem in s):
                return True
        return False

    def is_in_canoe(self, pt: Point, player: Player):
        """Return whether pt is in a canoe belonging to player"""
        if player == Player.red:
            moves = self.board.reds
        else:
            moves = self.board.yellows
        pt_idx = pt.to_idx()
        if not moves[pt_idx]:
            return False
        for s in self.solns_dict[pt_idx]:
            if all(moves[elem] for elem in s):
                return True
        return False

    def is_over(self) -> bool:
        """Return True iff game is over

        Also set self.winner to 0 for tie, 1 for red win, -1 for yellow win, or
        None for ongoing"""
        if self.last_move is None:
            return False
        if self.moves_made <= 2 * 7:  # requires 8+ moves to win
            return False
        if self.board.open_spaces <= 0:
            self.winner = 0
            return True
        if self.current_player.other == Player.red:
            # apply_move just called by current_player.other
            moves = self.board.reds
        else:
            moves = self.board.yellows
        winner = self.current_player.other
        last_move_id = self.last_move.point.to_idx()
        new_canoe = False
        for s in self.solns_dict[last_move_id]:
            if all(moves[elem] for elem in s):
                new_canoe = True
                break
        if new_canoe:
            canoes = []
            for s in self.solns:
                if all(moves[elem] for elem in s):
                    canoes.append(s)
            if len(canoes) >= 2:
                for c1 in canoes:
                    for c2 in canoes:
                        if not any(cc in c2 for cc in c1):
                            self.winner = winner
                            self.winning_canoes = [c1, c2]
                            return True
        return False

    def legal_moves(self) -> list[Move]:
        """Get a list of all legal moves"""
        return [Move.play(pt) for pt in self.board.return_open_spaces()]

    def is_valid_move(self, move):
        """Given a move, ensure it is valid (available and on grid)"""
        if self.is_over():
            return False
        return self.board.get(move.point) is None and self.board.is_on_grid(move.point)

    @classmethod
    def new_game(cls, first_player: Player = Player.red):
        """Start a new game with a new board"""
        board = Board()
        return GameState(board, first_player, None)

    @property
    def moves_made(self):
        return 61 - self.board.open_spaces


def print_error(msg):
    """Print a color-formatted error message"""
    print(f"\033[92m{msg}\033[0m")


def format_colored(turn, msg):
    """Print a color-formatted message"""
    if turn == Player.red:
        return f"\033[91m{msg}\033[0m"
    elif turn == Player.yellow:
        return f"\033[93m{msg}\033[0m"
    else:
        return f"{msg}"
