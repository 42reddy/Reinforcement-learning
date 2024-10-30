from main import Trader
import numpy as np
import matplotlib.pyplot as plt

actions = np.array([-1, 0, 1])
n_shares = 10
gamma = 0.95
instance = Trader(actions)
data = instance.get_data(start_date='2020-01-01', end_date='2024-09-01', ticker='MSFT')['Close'].pct_change().dropna().to_numpy()*100
"""for i in range(len(data)):
    data[i] = (data[i] - min(data)) / (max(data) - min(data))"""
data1 = data[np.argwhere(data > 0)] / np.abs(max(data))
data2 = data[np.argwhere(data < 0)] / np.abs(min(data))
data = np.concatenate([data1, data2])
plt.hist(data, bins=200)
episodes = []
episode_length = 100
for i in range(int(len(data)/episode_length)):
    episodes.append(data[i:i+episode_length])

model = instance.build_Q()
model.compile('adam', loss='mse')

for i in range(len(data)-31):
    state = np.array(data[i : i + 30])
    state = state.reshape((1,30,1))
    q_values = np.array(model.predict(state))
    action = instance.action(actions, q_values, 0.1)
    ind = np.argwhere(actions == action)
    next_state = data[i + 31]
    reward = (1 + action * next_state)  # * (state / data[0])
    n_s = data[i+1 : i+31]
    n_s = n_s.reshape((1,30,1))
    q_target = reward + gamma * (np.max(model.predict(n_s)))
    q_f = q_values
    q_f[0][ind] = q_target
    target = q_f
    model.fit(np.array(state), target, epochs=1)

