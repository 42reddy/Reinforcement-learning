from main import Trader
from main import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt

actions = [-1, 0, 1]
n_shares = 10
gamma = 0.95

instance = Trader(actions)
data = instance.get_data(start_date='2020-01-01', end_date='2024-09-01', ticker='MSFT')['Close'].pct_change().dropna().to_numpy()*100
for i in range(len(data)):
    data[i] = (data[i] - min(data)) / (max(data) - min(data))

plt.hist(data, bins=200)
episodes = []
episode_length = 100
for i in range(int(len(data)/episode_length)):
    episodes.append(data[i:i+episode_length])

model = instance.build_Q()
model.compile('adam', loss='mse')
Q_target = instance.build_Q()
Q_target.compile('adam', loss='mse')
exp_buffer = ExperienceReplay(1000)
batch_size = 32

for i in range(len(data)-31):
    state = np.array(data[i: i + 30])   # current state
    state = state.reshape(1, 30, 1)     # reshape for input dimensions
    q_values = model.predict(state)[0]  # predict q values at current state
    action = instance.action(actions, q_values, 0.1)   # take action based on q values
    ind = actions.index(action)
    next_state = data[i + 31]           # get next state
    reward = instance.reward_function(action, next_state)  # calculate reward
    exp = (state, ind, reward, data[i+1: i+31].reshape((1,30,1)))
    exp_buffer.add_experience(exp)

    if len(exp_buffer.buffer) >= batch_size:
        exps = exp_buffer.sample_experiences(batch_size)
        q_values = np.zeros((32,3))
        s = np.zeros((32,30,1))

        for j, (s_j, a, r, ns) in enumerate(exps):

            s[j] = s_j
            q_values[j] = model.predict(s_j)[0]
            q_target = q_values[j][a] + gamma * (r + (np.max(Q_target.predict(ns)[0]) - q_values[j][a]))
            q_values[j][a] = q_target

        model.fit(s, q_values, epochs=10)  # fit the model.





