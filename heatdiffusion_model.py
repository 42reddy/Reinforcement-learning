import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

class ExperienceReplay:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

class Agent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = 1

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(self.state_size,)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer='adam')
        return model

    def action(self, actions, q_values, epsilon):
        if np.random.rand() <= epsilon:
            return random.choice(actions)
        else:
            return actions[np.argmax(q_values)]

    def calculate_reward(self, action, grid):
        variance = np.mean(grid ** 2)
        grid[action] -= 1  # Cool the selected point
        new_variance = np.mean(grid ** 2)
        new_grid = grid.copy()
        for i in range(1, self.state_size - 1):
            new_grid[i] = grid[i] + self.alpha * (new_grid[i - 1] + new_grid[i + 1] - 2 * new_grid[i])

        reward = variance - new_variance
        return reward, new_grid

    def train_step(self, experiences, target_model):
        states, actions, rewards, next_states = experiences
        target_q_values = target_model.predict(next_states,verbose=0)
        target_q_values = rewards + gamma * np.amax(target_q_values, axis=1)
        target_q_values = target_q_values.reshape(-1, 1)

        mask = tf.one_hot(actions, self.action_size)
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values = tf.reduce_sum(q_values * mask, axis=1)
            q_values = tf.expand_dims(q_values, axis=1)  # Add dimension to match target_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

class heat_diffusion():
    def __init__(self):
        self.grid_size = 50
        self.grid = np.zeros(self.grid_size)
        self.grid[self.grid_size // 2] = 100  # Heat source at center
        self.alpha =1

    def step(self, action):
        variance = np.var(self.grid)
        self.grid[action] -= 1  # Cool the selected point

        new_variance = np.var(self.grid)
        reward = variance - new_variance

        new_grid = self.grid.copy()
        for i in range(1, self.grid_size - 1):
            new_grid[i] = new_grid[i] + self.alpha * (new_grid[i - 1] + new_grid[i + 1] - 2 * new_grid[i])

        self.grid = new_grid

        return self.grid.copy(), reward*100


# --- Main Training Loop ---


if __name__ == "__main__":
    # Hyperparameters
    grid_size = 50
    timesteps = 500
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.95
    batch_size = 32
    buffer_size = 10000
    learning_rate = 0.001
    num_episodes = 20
    update_target_freq = 20

    # Initialize
    env = heat_diffusion()
    state_size = env.grid_size
    action_size = env.grid_size
    agent = Agent(state_size, action_size)
    target_model = agent.build_model()
    target_model.set_weights(agent.model.get_weights())
    replay_buffer = ExperienceReplay(buffer_size)

    for episode in range(num_episodes):
        state = env.grid.copy()
        total_reward = 0
        x = 0
        loss1 = 0
        for t in range(timesteps):
            # Choose action
            q_values = agent.model.predict(state.reshape(1, -1),verbose=0)[0]
            action = agent.action(list(range(action_size)), q_values, epsilon)
            x += np.var(state)
            # Take action, observe reward and next state
            next_state, reward = env.step(action)
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.add(state, action, reward, next_state)

            # Train the agent
            if len(replay_buffer.buffer) >= batch_size:
                experiences = replay_buffer.sample(batch_size)
                loss = agent.train_step(experiences, target_model)
                loss1+= loss
            # Update state
            state = next_state

            # Decrement epsilon
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

        # Update target model
        if episode % update_target_freq == 0:
            target_model.set_weights(agent.model.get_weights())

        print(f"Episode: {episode}, Total Reward: {total_reward},Varience: {x}, loss:{loss1}")