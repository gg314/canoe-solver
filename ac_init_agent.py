'''
ac-init-agent.py: initialize a keras model for a Canoe agent
'''

import argparse
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, ZeroPadding2D
from tensorflow.keras.models import Model
from canoebot import encoders
from canoebot import utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', '-o', required=False, default="ac-v1")
  args = parser.parse_args()
  output_file = args.output
  encoder = encoders.ExperimentalEncoder() # or RelativeEncoder

  board_input = Input(shape=encoder.shape(), name='board_input')

  conv1 = Conv2D(128, (7, 7), padding='same', data_format='channels_first', activation='relu')(board_input)
  conv2 = Conv2D(128, (5, 5), padding='same', data_format='channels_first', activation='relu')(conv1)
  conv3 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu')(conv2)
  conv4 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu')(conv3)
  conv5 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu')(conv4)
  conv6 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu')(conv5)
  conv7 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu')(conv6)
  conv8 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu')(conv7)
  conv9 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu')(conv8)
  flat = Flatten()(conv9)
  processed_board = Dense(512)(flat)
  policy_hidden_layer = Dense(512, activation='relu')(processed_board)
  policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)
  value_hidden_layer = Dense(512, activation='relu')(processed_board)
  value_output = Dense(1, activation='tanh')(value_hidden_layer)

  model = Model(inputs=board_input, outputs=[policy_output, value_output])
  model.summary()

  utils.save_model(model, output_file)
   

if __name__ == '__main__':
  main()
