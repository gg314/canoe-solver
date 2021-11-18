# Based on Deep Learning and the Game of Go, Chapter 6 (2018, Manning)

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
import numpy as np
import sys
import encoders


np.random.seed(123)
X = np.load('./generated_games/features.npy')
Y = np.load('./generated_games/labels.npy')
samples = X.shape[0]

encoder = encoders.OnePlaneEncoder()
input_shape = (encoder.board_height, encoder.board_width, encoder.num_planes )
X = X.reshape(samples, encoder.board_height, encoder.board_width, encoder.num_planes )
# Y = Y.reshape(samples, board_size)

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

print(X.shape)
print(Y.shape)

adadelta = Adadelta()
model = Sequential()
model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Dropout(rate=0.125))
model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.125))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.125))
model.add(Dense(6*13, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = 64, epochs = 15, verbose = 1, validation_data = (X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose = 0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("./generated_models/test1.h5")

test_board = np.array([[
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  
    0,  0,  0,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  
    0, -1, -1, -1, -1, -1, -1,  0,  1,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  
]])

test_board = test_board.reshape(1, encoder.board_height, encoder.board_width, encoder.num_planes)

move_probs = model.predict(test_board)[0]
print(move_probs)
i = 0
for row in range(encoder.board_height):
    row_formatted = []
    for col in range(encoder.board_width):
        row_formatted.append(f"{move_probs[i]:.3f}")
        i += 1
    print(' '.join(row_formatted))