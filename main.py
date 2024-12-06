import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import random
from collections import deque


class ExperienceReplay:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_experiences(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))


class Trader():

    def __init__(self, actions):

        self.actions = actions

    def reward_function(self, action, p):

        return action * p

    def build_Q(self):

        model = keras.Sequential()
        model.add(keras.layers.Conv1D(256, kernel_size=4, activation='relu', input_shape=(30, 2)))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.Conv1D(128, kernel_size=4, activation='relu'))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dense(2, activation=None))

        return model

    def DQN(self):

        model = keras.Sequential()
        model.add(keras.layers.Conv1D(16, 5))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.LSTM(200,activation='tanh', return_sequences=True))
        model.add(keras.layers.LSTM(50))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(2, activation=None))

        return model

    def get_data(self, start_date, end_date, ticker):

        return yf.download(ticker, start=start_date, end=end_date, interval='1d')

    def action(self, actions, q_values, epsilon):
        q_max = max(q_values)
        a_rest = np.delete(actions, np.argmax(q_max))
        a = np.random.uniform(0, 1)
        if a < epsilon:
            action = np.random.choice(a_rest)
        else:
            action = actions[np.argmax(q_max)]

        return action
