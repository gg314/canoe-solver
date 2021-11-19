import argparse
import sys
import numpy as np
import time
import agent
import encoders
from board import GameState, Player



def generate_game(rounds=None, max_moves=100):
    boards, moves = [], []
    encoder = encoders.OnePlaneEncoder()
    game = GameState.new_game()
    bot = agent.GreedyGreedyAgent()
    num_moves = 0

    while not game.is_over():
        move = bot.select_move(game)
        boards.append(encoder.encode(game))
        move_one_hot = np.zeros(encoder.num_points())
        move_one_hot[encoder.encode_point(move.point)] = 1
        moves.append(move_one_hot)
        game = game.apply_move(move)
        # game.print_board()
        num_moves += 1
        if num_moves > max_moves:
            break

    return np.array(boards), np.array(moves)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', '-r', type=int, default=100)
    parser.add_argument('--max-moves', '-m', type=int, default=100, help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--board-out', default="./generated_games/features.npy")
    parser.add_argument('--move-out', default="./generated_games/labels.npy")
    args = parser.parse_args()
    xs = []
    ys = []
    
    for i in range(args.num_games):
        print(f"Generating game {i+1}/{args.num_games}")
        x, y = generate_game(args.rounds, args.max_moves)
        xs.append(x)
        ys.append(y)
    
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    np.save(args.board_out, x)
    np.save(args.move_out, y)
    


if __name__ == '__main__':
    main()
