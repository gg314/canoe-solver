import argparse
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, ZeroPadding2D, concatenate
from tensorflow.keras.models import Model
import agent
import encoders
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', default="model-init")
    args = parser.parse_args()
    output_file = args.output_file
    encoder = encoders.RelativeEncoder()

    board_input = Input(shape=encoder.shape(), name='board_input')

    conv1a = ZeroPadding2D((2, 2))(board_input)
    conv1b = Conv2D(64, (5, 5), activation='relu')(conv1a)
    conv2a = ZeroPadding2D((1, 1))(conv1b)
    conv2b = Conv2D(64, (3, 3), activation='relu')(conv2a)
    conv3a = ZeroPadding2D((1, 1))(conv2b)
    conv3b = Conv2D(64, (3, 3), activation='relu')(conv3a)
    flat = Flatten()(conv3b)
    processed_board = Dense(512)(flat)
    
    policy_hidden_layer = Dense(512, activation='relu')(processed_board)
    policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)

    value_hidden_layer = Dense(512, activation='relu')(processed_board)
    value_output = Dense(1, activation='tanh')(value_hidden_layer)

    model = Model(inputs=board_input, outputs=[policy_output, value_output])
    model.summary()

    new_agent = agent.QAgent(model, encoder)
    utils.save_model(model, output_file)
    

if __name__ == '__main__':
    main()
