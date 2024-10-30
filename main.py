import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf


class Trader():

    def __init__(self,actions):

        self.actions = actions

    def reward_function(self, action, p_t, p_old, p_start):

        return (1 + action* (p_t-p_old) / p_old) * (p_old / p_start)

    def build_Q(self):

        model = keras.Sequential()
        model.add(keras.layers.Conv1D(256, kernel_size=4, activation='relu', input_shape=(30,1)))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.Conv1D(128,kernel_size=4, activation='relu'))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(200, activation='relu'))
        model.add(keras.layers.Dense(100,activation='relu'))
        model.add(keras.layers.Dense(50, activation= 'relu'))
        model.add(keras.layers.Dense(3, activation='softmax'))

        return model

    def get_data(self, start_date, end_date, ticker):

        return yf.download(ticker, start=start_date, end=end_date, interval='1d')

    def action(self,actions, q_values, epsilon):
        q_max = max(q_values)
        a_rest = np.delete(actions, np.argmax(q_max))
        a = np.random.uniform(0,1)
        if a < epsilon:
            action = np.random.choice(a_rest)
        else:
            action = actions[np.argmax(q_max)]

        return action


