import argparse
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, ZeroPadding2D, concatenate
from tensorflow.keras.models import Model
import agent
import encoders
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', default="qinit")
    args = parser.parse_args()
    output_file = args.output_file
    encoder = encoders.OnePlaneEncoder()

    board_input = Input(shape=encoder.shape(), name='board_input')
    action_input = Input(shape=(encoder.num_points(),), name='action_input')
    conv1a = ZeroPadding2D((2, 2))(board_input)
    conv1b = Conv2D(64, (5, 5), activation='relu')(conv1a)
    conv2a = ZeroPadding2D((1, 1))(conv1b)
    conv2b = Conv2D(64, (3, 3), activation='relu')(conv2a)
    flat00 = Flatten()(conv2b)
    processed_board = Dense(512)(flat00)
    board_and_action = concatenate([action_input, processed_board])
    hidden_layer = Dense(256, activation='relu')(board_and_action)
    value_output = Dense(1, activation='tanh')(hidden_layer)
    model = Model(inputs=[board_input, action_input], outputs=value_output)

    new_agent = agent.QAgent(model, encoder)
    utils.save_model(model, "./generated_models/" +  output_file + ".h5")
    

if __name__ == '__main__':
    main()
