'''
ac-train.py: use self-play experience to train a new iteration of the neural
  net; benchmark against previous iteration.
'''
import argparse
import h5py
from tensorflow.python.ops.gen_math_ops import exp

from canoebot import agent
from canoebot import encoders
from canoebot import utils
from canoebot.board import Player, GameState, format_colored
from canoebot.experience import combine_experience, load_experience, DataGenerator
import numpy as np
from scipy.stats import binom
import tensorflow as tf


def make_progress_bar(wins, trials, total, winner):
  total_bars = 100
  percent = int(total_bars*(wins/(trials)))
  pre = "="*percent
  post = "-"*(total_bars - percent)
  sq = format_colored(winner, "■")
  print("")
  print(f"{trials:3}/{total:3} {sq} |{pre}|{post}| ({(100 * wins / trials):.1f}%)", end="\r")


def main():
  ''' train NN on experience, benchmark vs. previous iteration '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-in', required=True)
#  parser.add_argument('--model-out', required=True)
  parser.add_argument('--learning-rate', type=float, default=0.000000011) # 00000005 # 000000006 = 50%
  parser.add_argument('--bs', type=int, default=2048)
  parser.add_argument('--benchmark-trials', type=int, default=500)
  parser.add_argument('-g', '--generator', help="use generator instead of loading into memory", action="store_true")
  parser.add_argument('-m', '--mute', help="suppress messages", action="store_true")
  parser.add_argument('experience', nargs='+')

  args = parser.parse_args()
  model_in_filename = "ac-v" + str(args.model_in)
  experience_files = args.experience
  model_out_filename = "ac-v" + str(int(args.model_in) + 1) # args.model_out
  benchmark_trials = args.benchmark_trials
  learning_rate = args.learning_rate
  batch_size = args.bs

  agent1 = agent.ACAgent(utils.load_model(model_in_filename), encoders.ExperimentalEncoder())

  if args.generator:
    dataset = tf.data.Dataset.from_generator(
      DataGenerator(experience_files),
      args=(batch_size,),
      output_signature=(
        tf.TensorSpec(shape=(None, 8, 6, 13), dtype=tf.int64),
        (tf.TensorSpec(shape=(None, 78), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64))
      ))

    agent1.generator_train(dataset, learning_rate=learning_rate)
  else:
    input("Run without generator?")
    buffers = []
    for exp_filename in experience_files:
      with h5py.File("./generated_experience/" + exp_filename + ".h5") as exp_file:
        exp_buffer = load_experience(exp_file)
        buffers.append(exp_buffer)
    all_experience = combine_experience(buffers)
    agent1.train(all_experience, learning_rate=learning_rate, batch_size=batch_size)

  utils.save_model(agent1.model, model_out_filename)

  wins = { "RL": 0, "pre": 0, "ties": 0 }
  agent2 = agent.ACAgent(utils.load_model(model_in_filename), encoders.ExperimentalEncoder())
  bots = {
    Player.red: agent1,
    Player.yellow: agent2,
  }

  for trial in range(benchmark_trials):
    if trial % 2:
      first_player = Player.red
    else:
      first_player = Player.yellow
    game = GameState.new_game(first_player)
    while not game.is_over():
      bot_move = bots[game.current_player].select_move(game)
      game = game.apply_move(bot_move)
    if game.winner == Player.red:
      wins['RL'] += 1
    elif game.winner == Player.yellow:
      wins['pre'] += 1
    else:
      wins['ties'] += 1
    # game.print_board()
    if not args.mute:
      make_progress_bar(wins['RL'], trial+1, benchmark_trials, game.winner)

  k = int(wins['RL'])
  wins['RL'] = wins['RL'] / benchmark_trials
  wins['pre'] = wins['pre'] / benchmark_trials
  wins['ties'] = wins['ties'] / benchmark_trials
  print(f"\r\nBrenchmark: {wins}")
  p_val = np.sum([binom.pmf(k, benchmark_trials, 0.5) for k in range(k, benchmark_trials+1)])
  print(f"p = {p_val:.3}")


if __name__ == '__main__':
  main()
