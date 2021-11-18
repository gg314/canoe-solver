import argparse

import h5py

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Flatten, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Sequential
import agent
import encoders
import utils

def layers(input_shape):
    return [
        ZeroPadding2D((3, 3), input_shape=input_shape),
        Conv2D(64, (7, 7), padding='valid'),
        Activation('relu'),

        ZeroPadding2D((2, 2)),
        Conv2D(64, (5, 5)),
        Activation('relu'),

        ZeroPadding2D((2, 2)),
        Conv2D(64, (5, 5)),
        Activation('relu'),

        ZeroPadding2D((2, 2)),
        Conv2D(48, (5, 5)),
        Activation('relu'),

        ZeroPadding2D((2, 2)),
        Conv2D(48, (5, 5)),
        Activation('relu'),

        ZeroPadding2D((2, 2)),
        Conv2D(32, (5, 5)),
        Activation('relu'),

        ZeroPadding2D((2, 2)),
        Conv2D(32, (5, 5)),
        Activation('relu'),

        Flatten(),
        Dense(1024),
        Activation('relu'),
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file')
    args = parser.parse_args()
    output_file = args.output_file

    encoder = encoders.OnePlaneEncoder()
    model = Sequential()
    for layer in layers(encoder.shape()):
        model.add(layer)
    model.add(Dense(encoder.num_points()))
    model.add(Activation('softmax'))
    new_agent = agent.PolicyAgent(model, encoder)

    utils.save_model(model, "init")
    

if __name__ == '__main__':
    main()
