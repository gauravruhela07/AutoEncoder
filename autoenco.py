import numpy as np
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers
import os.path
import math

def sigmoid(arr):
    return 1 / (1 + np.exp(-np.asarray(arr)))

class nn:
    def __init__(self):
        self.m = None
        self.encoder = None
        self.decoder = None
        self.initialize_nn()

    def initialize_encoder(self):
        loc = Sequential()
        loc.add(Dense(units=400, activation='tanh', input_dim=784, kernel_initializer='random_uniform', bias_initializer='zeros'))
        loc.add(Dense(units=150, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
        loc.add(Dense(units=10, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))
        return loc

    def initialize_decoder(self):
        loc = Sequential()
        loc.add(Dense(units=150, activation='tanh', input_dim=10, kernel_initializer='random_uniform', bias_initializer='zeros'))
        loc.add(Dense(units=400, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
        loc.add(Dense(units=784, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))
        return loc

    def initialize_nn(self):
        # encoder + decoder or from file
        if os.path.isfile('encoder.h5') and os.path.isfile('decoder.h5'):
            self.encoder = load_model('encoder.h5')
            self.decoder = load_model('decoder.h5')
        else:
            self.encoder = self.initialize_encoder()
            self.encoder.compile(optimizer=optimizers.Adam(lr=0.0002), loss='mean_squared_error', metrics=['accuracy'])
            self.decoder = self.initialize_decoder()
            self.decoder.compile(optimizer=optimizers.Adam(lr=0.0002), loss='mean_squared_error', metrics=['accuracy'])

        self.m = Sequential()
        self.m.add(self.encoder)
        self.m.add(self.decoder)
        self.m.compile(optimizer=optimizers.Adam(lr=0.0002), loss='mean_squared_error', metrics=['accuracy'])

    def save_all(self):
        self.encoder.save('encoder.h5')
        self.decoder.save('decoder.h5')
        del self.encoder
        del self.decoder
        self.initialize_nn()

if __name__ == '__main__':
    mode = input('Please select a mode: ')

    if mode == 'train':
        while True:
            b = False

            mnist_arr = []
            mnist_x = []
            network = nn()

            with open('data/mnist_train.csv', 'r') as csv_file:
                for row in csv.reader(csv_file, delimiter=','):
                    if b:
                        img = row[1:]
                        img = np.array(img, dtype='float64')
                        mnist_x.append(img)
                        mnist_arr.append(sigmoid(img))
                    else:
                        b = True

                    if len(mnist_arr) == 40:
                        network.m.fit(np.asarray(mnist_x), np.asarray(mnist_arr), batch_size=len(mnist_arr), epochs=10, verbose=1)
                        mnist_arr = []
                        mnist_x = []

                print('Saving...')
                network.save_all()
    elif mode == 'custom':
        network = nn()

        while True:
            nums = input('Provide your ten parametric numbers. -- bounds (0, 1) ').split(' ')
            if len(nums) == 10:
                out = network.decoder.predict(np.asarray([nums], dtype='float64'), batch_size=1)
                out = out.reshape(28, 28)

                plt.title('Decoded vector output')
                plt.imshow(out, cmap='gray')
                plt.show()
            else:
                print('Wrong number of args! Please input *exactly* ten...')
    elif mode == 'testing':
        network = nn()

        b = False

        with open('data/mnist_train.csv', 'r') as csv_file:
            for row in csv.reader(csv_file, delimiter=','):
                if b:
                    img = row[1:]
                    img = np.array(img, dtype='float64')

                    out = network.m.predict(np.asarray([img], dtype='float64'), batch_size=1)
                    out = out.reshape(28, 28)

                    plt.title('Test in/out')
                    plt.imshow(out, cmap='gray')
                    plt.show()
                else:
                    b = True
    elif mode == 'random':
        network = nn()

        while True:
            gen = network.decoder.predict(np.random.rand(1, 10), batch_size=1)
            gen = gen.reshape(28, 28)

            plt.title('Sample of Random Image Generation')
            plt.imshow(gen, cmap='gray')
            plt.show()
