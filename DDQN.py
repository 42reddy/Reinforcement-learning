from main import Trader
from main import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
import yfinance as yf

actions = [-1, 1]
n_shares = 10
gamma = 0.9
epsilon = 0.1
decay_rate = 0.95

instance = Trader(actions)
data = instance.get_data(start_date='2020-01-01', end_date='2024-09-01', ticker='MSFT')['Close'].pct_change().dropna().to_numpy()*100
volume = yf.download(tickers="MSFT", start='2020-01-01', end='2024-09-01')['Volume'].to_numpy()

data = (data - np.mean(data)) / np.std(data)
volume = (volume - np.mean(volume)) / np.std(volume)

# plt.hist(data, bins=1000, density=True)


model = instance.DQN()                  # nn to predict q_values
model.compile(Adam(0.001), loss='mse')
exp_buffer = ExperienceReplay(1000)         # experience replay
batch_size = 40

Q_target = instance.DQN()               # target model
Q_target.compile(Adam(0.001), loss='mse')

input_shape = (30, 2)  # Adjust to match your input data shape
model.build(input_shape=(None, *input_shape))
Q_target.build(input_shape=(None, *input_shape))


for i in range(len(data) - 31):

    state = np.hstack((data[i : i+30], volume[i : i+30])).reshape(1, 30, 2)  # Initial state
    qvalues = model.predict(state, verbose=0)  # Predict Q-values
    epsilon = min(float(0.0001), epsilon * decay_rate)
    action = instance.action(actions, qvalues, epsilon=0.01)  # Select action
    action_idx = actions.index(action)         # action index
    reward = instance.reward_function(action, 10*data[i+31])    # calculate reward
    print(reward)
    next_state = np.hstack((data[i+1 : i+31], volume[i+1 : i+31])).reshape(1, 30, 2)  # Next state
    experience = (state, action_idx, reward, next_state)          # gather experiences
    exp_buffer.add_experience(experience)

    """
    sample random sample from the buffer and update the Q values"""
    if len(exp_buffer.buffer) >= 400:
        exps = exp_buffer.sample_experiences(batch_size)
        s = np.zeros((batch_size, 30, 2))
        q_values = np.zeros((batch_size, len(actions)))

        for j, (s_j, a, r, ns) in enumerate(exps):
            s[j] = s_j
            q_values[j] = model.predict(s_j, verbose=0)
            # best_action = np.argmax(model.predict(ns, verbose=0)[0])  # Use online network to find the best action
            q_target = r + gamma * np.argmax(model.predict(ns, verbose=0)[0])  # Evaluate using target network

            q_values[j][a] = q_target

        model.fit(s, q_values, epochs=1)

    if i % 60 == 0:
        print('hey')
        Q_target.set_weights(model.get_weights())


