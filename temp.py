''' temporary scripts, delete later '''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from canoebot.board import make_solns, Player
from canoebot.experience import ExperienceBuffer, combine_experience, load_experience, DataGenerator

def fmt(id):
  r = id // 13
  c = id % 13
  return f"{{{c}, {r}}}"


def print_elixir_canoes():
  _solns, solns_dict = make_solns()
  for r in range(0, 6):
    print("[ ", end="")
    for c in range(0, 13):
      val = r*13+c
      part_canoes = solns_dict[val]
      print("[", end="")
      for pc in part_canoes:
        strs = [ fmt(p) for p in pc ] + [ fmt(val) ]
        print("{", end="")
        print(", ".join(strs), end="")
        print("}, ", end="")
      print("]", end=", ")
    print("], ", end="")


def split_data():
  experience_files = ["e5-1"]
  buffers = []
  for exp_filename in experience_files:
    exp_buffer = load_experience(h5py.File("./generated_experience/" + exp_filename + ".h5"))
    buffers.append(exp_buffer)
  exp = combine_experience(buffers)
  
  print(len(exp.states))
  print(len(exp.actions))
  print(len(exp.rewards))
  print(len(exp.advantages))
  keep = int(270/323.0 * 877428)
  print(keep)

  long = ExperienceBuffer(exp.states[0:keep], exp.actions[0:keep], exp.rewards[0:keep], exp.advantages[0:keep])
  short = ExperienceBuffer(exp.states[keep:], exp.actions[keep:], exp.rewards[keep:], exp.advantages[keep:])

  with h5py.File("./generated_experience/e5-l.h5", 'w') as experience_outf:
    long.serialize(experience_outf)
  with h5py.File("./generated_experience/e5-s.h5", 'w') as experience_outf:
    short.serialize(experience_outf)

def make_progress_bar(wins, trials, total, winner):
  total_bars = 100
  percent = int(total_bars*(wins/(trials+1)))
  pre_bars = "-"*percent
  post_bars = "-"*(total_bars - percent)
  print(f"{trials + 1:3}/{total:3} |{pre_bars}|{post_bars}| ({(100 * wins / (trials + 1)):.1f}%)")

def make_img():
  num_moves = 13*6
  move_probs = np.random.rand(num_moves)
  estimated_value = 0.123

  # for rr in range(6):
  #   for cc in range(13):
  #     print(f"{move_probs[13*rr + cc]:.3f} ", end="")
  #   print(" ")
  dummy = np.ones(num_moves)
  zzzzzz = [0, 3, 4, 5, 6, 7, 8, 9, 12, 52, 64, 65, 66, 67, 75, 76, 77]
  for idx in zzzzzz:
    dummy[idx] = 0

  eps = 1e-6
  move_probs = np.multiply(dummy, move_probs)
  move_probs = np.clip(move_probs, eps, 1 - eps)
  move_probs = move_probs / np.sum(move_probs)
  candidates = np.arange(num_moves)
  ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)

  for point_idx in ranked_moves:
    if point_idx not in zzzzzz:
      for idx in zzzzzz:
        move_probs[idx] = np.nan
      heatmap = move_probs.reshape((6, 13))
      movemap = heatmap
      fig, ax = plt.subplots(2)
      im = ax[0].imshow(heatmap, cmap='viridis')
      im = ax[1].imshow(movemap, cmap='hot')
      ax[0].spines['top'].set_visible(False)
      ax[0].spines['right'].set_visible(False)
      ax[0].spines['bottom'].set_visible(False)
      ax[0].spines['left'].set_visible(False)
      ax[0].set_axis_off()
      ax[1].set_axis_off()
      ax[0].set_title(f"AAA to move\nChosen moves: {ranked_moves[0:6]}\nActual move: {point_idx}\nEstimated value: {estimated_value}")
      for i in range(6):
          for j in range(13):
              if (13*i+j) not in zzzzzz:
                _text = ax[0].text(j, i, 13*i+j, ha="center", va="center", color="w")
      fig.tight_layout()
      plt.show()


def make_heatmap(arr, title):
  num_moves = 13*6
  move_probs = arr
  dummy = np.ones(num_moves)
  zzzzzz = [0, 3, 4, 5, 6, 7, 8, 9, 12, 52, 64, 65, 66, 67, 75, 76, 77]
  for idx in zzzzzz:
    dummy[idx] = 0
  dummy = dummy.reshape((6, 13))

  eps = 1e-6
  move_probs = move_probs / np.sum(move_probs)
  move_probs = np.clip(move_probs, eps, 1 - eps)
  move_probs = np.multiply(dummy, move_probs)
  move_probs[move_probs == 0] = 'nan'

  heatmap = move_probs
  fig, ax = plt.subplots(1)
  im = ax.imshow(heatmap, cmap='viridis', vmin=0)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.set_axis_off()
  ax.set_title(title + "\n", fontsize=16, color='#ce9178', fontfamily='DINPro') # d4d4d4
  # for i in range(6):
  #     for j in range(13):
  #         if (13*i+j) not in zzzzzz:
  #           _text = ax[0].text(j, i, 13*i+j, ha="center", va="center", color="w")
  fig.patch.set_facecolor('#1e1e1e')
  fig.tight_layout()
  plt.show()


def test_generator(filenames):
  dg = DataGenerator(filenames)
  for v in dg:
    print(v)

if __name__ == "__main__":
  arr = np.array([[0.000, 0.736, 0.714, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.768, 0.738, 0.000],
                  [0.724, 0.732, 0.750, 0.728, 0.748, 0.744, 0.738, 0.740, 0.748, 0.724, 0.714, 0.746, 0.704],
                  [0.732, 0.714, 0.746, 0.746, 0.760, 0.762, 0.758, 0.738, 0.752, 0.756, 0.764, 0.730, 0.736],
                  [0.712, 0.730, 0.736, 0.748, 0.772, 0.698, 0.740, 0.742, 0.736, 0.752, 0.736, 0.714, 0.738],
                  [0.000, 0.720, 0.726, 0.728, 0.722, 0.718, 0.738, 0.732, 0.738, 0.764, 0.740, 0.754, 0.000],
                  [0.000, 0.000, 0.000, 0.722, 0.704, 0.728, 0.762, 0.736, 0.738, 0.736, 0.000, 0.000, 0.000]])
  make_heatmap(arr, "Move heatmap\nrandom agent v. 1")
  arr = np.array([[0.000, 0.040, 0.756, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.410, 0.062, 0.000],
                  [0.042, 0.062, 0.064, 0.812, 0.880, 0.050, 0.066, 0.434, 0.048, 0.494, 0.576, 0.064, 0.872],
                  [0.064, 0.060, 0.676, 0.076, 0.896, 0.822, 0.956, 0.274, 0.066, 0.102, 0.656, 0.838, 0.068],
                  [0.936, 0.040, 0.072, 0.848, 0.070, 0.740, 0.856, 0.430, 0.714, 0.706, 0.052, 0.052, 0.054],
                  [0.000, 0.052, 0.262, 0.052, 0.058, 0.072, 0.352, 0.062, 0.874, 0.072, 0.070, 0.046, 0.000],
                  [0.000, 0.000, 0.000, 0.732, 0.044, 0.250, 0.202, 0.498, 0.844, 0.790, 0.000, 0.000, 0.000]])
  make_heatmap(arr, "Move heatmap\nv. 2 (vs. random agent)")

  arr = np.array([[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004, 0.000, 0.000],
                  [0.006, 0.000, 0.020, 0.002, 0.006, 0.826, 0.006, 0.334, 0.482, 0.214, 0.418, 0.002, 0.000],
                  [0.002, 0.338, 0.542, 0.180, 0.550, 0.476, 0.538, 0.618, 0.546, 0.688, 0.430, 0.000, 0.002],
                  [0.000, 0.002, 0.296, 0.694, 0.498, 0.520, 0.618, 0.900, 0.660, 0.984, 0.452, 0.348, 0.004],
                  [0.000, 0.000, 0.332, 0.378, 0.744, 0.152, 0.550, 0.616, 0.378, 0.526, 0.506, 0.000, 0.000],
                  [0.000, 0.000, 0.000, 0.002, 0.270, 0.000, 0.006, 
                  0.690, 0.006, 0.260, 0.000, 0.000, 0.000]])
  make_heatmap(arr, "Move heatmap\nv. 11 (vs. random agent)")
  arr = np.array([[0.000, 0.062, 0.060, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.062, 0.056, 0.000],
                  [0.046, 0.062, 0.172, 0.046, 0.056, 0.538, 0.062, 0.450, 0.466, 0.356, 0.444, 0.052, 0.050],
                  [0.048, 0.424, 0.474, 0.352, 0.510, 0.478, 0.480, 0.520, 0.468, 0.518, 0.434, 0.058, 0.070],
                  [0.064, 0.056, 0.442, 0.434, 0.472, 0.530, 0.498, 0.516, 0.478, 0.480, 0.406, 0.382, 0.054],
                  [0.000, 0.050, 0.452, 0.388, 0.514, 0.324, 0.470, 0.472, 0.416, 0.476, 0.490, 0.050, 0.000],
                  [0.000, 0.000, 0.000, 0.066, 0.402, 0.066, 0.062, 0.514, 0.050, 0.388, 0.000, 0.000, 0.000]])
  make_heatmap(arr, "Move heatmap\nv. 11 (vs. self)")