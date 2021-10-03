import sys
import numpy as np



class RandomStrategy:
    def __init__(self):
        pass

    def make_move(self, board, previous_move_row=None, previous_move_col=None):
        return np.random.choice(return_open_spaces(board, len(board), len(board[0])))

class GreedyStrategy:
    def __init__(self):
        pass

    def make_move(self, board, previous_move_row=None, previous_move_col=None):
        print(f"Previous move: {previous_move_row}, {previous_move_col}")
        if previous_move_row == None:
            return np.random.choice(return_open_spaces(board, len(board), len(board[0])))
        
        open_spaces = []
        for r in [previous_move_row - 1, previous_move_row, previous_move_row + 1]:
            for c in [previous_move_col - 1, previous_move_col, previous_move_col + 1]:
                try:
                    if (board[r, c, 0]) > 0 and (board[r, c, 1] == 0):
                        open_spaces.append(board[r, c, 0])
                except: pass
        if len(open_spaces) > 0:
            return np.random.choice(open_spaces)
        else:
            return np.random.choice(return_open_spaces(board, len(board), len(board[0])))

class IndexError(Exception):
    pass


def return_open_spaces(board, rows, cols):
    open_spaces = []
    for r in range(rows):
        for c in range(cols):
            if board[r, c, 1] == 0:
                open_spaces.append(board[r, c, 0])
    return open_spaces
    
class CanoeGame:
    def __init__(self):
        self.new_game()
        self.strategy = GreedyStrategy()
    
    def next_turn(self):
        self.turn = self.turn % 2 + 1

    def new_game(self, rows = 6, cols = 13):
        self.turn = np.random.randint(2) + 1
        self.team1 = []
        self.team2 = []
        self.rows = rows
        self.cols = cols
        self.previous_move_row = None
        self.previous_move_col = None

        board = np.zeros((rows, cols, 2), dtype=np.int8)
        empties = [(0, 0), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 12), (4, 0), (4, 12), (5, 0), (5, 1), (5, 2), (5, 10), (5, 11), (5, 12)]
        for e in empties:
            board[e[0]][e[1]][1] = -1

        count = 1
        for row in range(rows):
            for col in range(cols):
                if board[row][col][1] != -1:
                    board[row][col][0] = count
                    count += 1
        self.unused_spaces = count - 1

        solns = []
        for r in range(rows - 1): # \__/
            for c in range(cols - 3):
                if -1 not in (board[r, c, 1], board[r, c+3, 1], board[r+1, c+1, 1], board[r+1, c+2, 1]):
                    solns.append( (board[r, c, 0], board[r, c+3, 0], board[r+1, c+1, 0], board[r+1, c+2, 0]) )
        for r in range(rows - 1): # /~~\
            for c in range(cols - 3):
                if -1 not in (board[r, c+1, 1], board[r, c+2, 1], board[r+1, c, 1], board[r+1, c+3, 1]):
                    solns.append( (board[r, c+1, 0], board[r, c+2, 0], board[r+1, c, 0], board[r+1, c+3, 0]) )
        for r in range(rows - 3): # (
            for c in range(cols - 1):
                if -1 not in (board[r, c+1, 1], board[r+1, c, 1], board[r+2, c, 1], board[r+3, c+1, 1]):
                    solns.append( (board[r, c+1, 0], board[r+1, c, 0], board[r+2, c, 0], board[r+3, c+1, 0]) )
        for r in range(rows - 3): # )
            for c in range(cols - 1):
                if -1 not in (board[r, c, 1], board[r+1, c+1, 1], board[r+2, c+1, 1], board[r+3, c, 1]):
                    solns.append( (board[r, c, 0], board[r+1, c+1, 0], board[r+2, c+1, 0], board[r+3, c, 0]) )

        self.board = board
        self.solns = solns

    def make_move(self, r, c):
        self.board[row, col, 1] = self.turn
        if self.turn == 1:
            self.team1.append(self.board[row, col, 0])
            solutions = self.check_solutions(self.team1)
        else:
            self.team2.append(self.board[row, col, 0])
            solutions = self.check_solutions(self.team2)
        if solutions == 2:
            self.print_board()
            print_colored(self.turn, f"Player {self.turn} wins!")
            self.new_game()
        else: # more messages here
            self.unused_spaces -= 1
            if self.unused_spaces == 0:
                print_colored(0, "No spaces left. It's a draw!")
                self.new_game()
            else:
                self.previous_move_row = row
                self.previous_move_col = col
                self.next_turn()


    def check_solutions(self, moves):
        canoes = []
        for s in self.solns:
            if all(elem in moves for elem in s):
                canoes.append(s)
        
        if len(canoes) >= 2:
            for c1 in canoes:
                for c2 in canoes:
                    if not any(cc in c2 for cc in c1):
                        return 2
        elif len(canoes) >= 1:
            return 1
        else:
            return 0


    def print_board(self):
        # print("1:", self.team1)
        # print("2:", self.team2)
        print("")
        for r in self.board:
            for c in r:
                if c[1] == 0:
                    print(f"{c[0]:02d}\033[0m", end=" ")
                elif c[1] == 1:
                    print(f"\033[91m ■\033[0m", end=" ")
                elif c[1] == 2:
                    print(f"\033[93m ■\033[0m", end=" ")
                else:
                    print("  ", end=" ")
            print("")
        print("")


def print_error(msg):
    print(f"\033[92m{msg}\033[0m")
    
def print_colored(turn, msg):
    if turn == 0:
        print(f"{msg}")
    elif turn == 1:
        print(f"\033[91m{msg}\033[0m")
    elif turn == 2:
        print(f"\033[93m{msg}\033[0m")





canoe = CanoeGame()

while True:
    
    canoe.print_board()

    if canoe.turn == 1:
        ai_move = canoe.strategy.make_move(canoe.board, canoe.previous_move_row, canoe.previous_move_col)
        index = np.where(canoe.board[:,:,0] == ai_move)
        try:
            row = index[0][0]
            col = index[1][0]
            if canoe.board[row, col, 0] <= 0:
                raise Exception("Invalid index")
            if canoe.board[row, col, 1] != 0:
                raise IndexError
            
            canoe.make_move(row, col)
        except Exception as e:
            print(e)
            sys.exit(f"Illegal AI move {index}")
            

    else:
        turn_str = "Select input \033[91m■\033[0m:\r\n" if (canoe.turn == 1) else "Select input \033[93m■\033[0m:\r\n"
        selection = input(turn_str)

        if selection == 'q':
            break
        else:
            index = np.where(canoe.board[:,:,0] == int(selection))
            try:
                row = index[0][0]
                col = index[1][0]
                if canoe.board[row, col, 0] <= 0:
                    raise Exception("Invalid index")
                if canoe.board[row, col, 1] != 0:
                    raise IndexError
                
                canoe.make_move(row, col)
            
            except IndexError:
                print_error("This index is occupied. Please try again.")

            except Exception as e:
                print_error("Invalid index. Please try again.")
                print(e)

