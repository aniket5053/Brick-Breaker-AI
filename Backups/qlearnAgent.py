import os
import pickle
import numpy as np
from breakout import Breakout

# Settings
NUM_EPISODES = 5000
LOG_INTERVAL = 500
STATE_DIM = 5
ACTION_SPACE = [0, 1, 2]
PICKLE_PATH = 'q_learning_data.pkl'


# --- Q-Learning Agent ---
class QLearnAgent:
    def __init__(self, state_dim, num_actions, alpha=0.1, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        self.q_table = {}  # {state_tuple: np.array[action_values]}
        self.rewards = []
        self.episode = 0

    def encode(self, state):
        return tuple(int(s) for s in state)

    def select_action(self, state):
        state = self.encode(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return np.argmax(self.q_table[state])  # Pure exploitation

    def update(self, state, action, reward, next_state):
        s = self.encode(state)
        s_ = self.encode(next_state)

        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.num_actions)
        if s_ not in self.q_table:
            self.q_table[s_] = np.zeros(self.num_actions)

        old_value = self.q_table[s][action]
        next_max = np.max(self.q_table[s_])
        self.q_table[s][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

    def remember_reward(self, total_reward):
        self.rewards.append(total_reward)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.q_table, self.rewards, self.episode), f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_table, self.rewards, self.episode = pickle.load(f)
            print(f"Loaded Q-table from episode {self.episode}")
        else:
            print("No saved data found. Starting fresh.")


# --- Reward Function ---
def calculate_reward(prev_bricks, prev_hits, game, done):
    reward = 0
    new_hits = game.get_number_paddle_hits_round() - prev_hits
    new_bricks = game.get_bricks_broken_attempt() - prev_bricks

    if new_hits > 0:
        reward += 1 + 3 * new_bricks
    reward += 2 * new_bricks

    if done:
        if game.get_isRoundLost():
            reward -= 100
        elif not any(b["breakable"] for b in game.bricks):
            reward += 50 + 200

    return reward


# --- Main Training Loop ---
def train():
    env = Breakout(log_to_csv=True)
    agent = QLearnAgent(state_dim=STATE_DIM, num_actions=len(ACTION_SPACE))
    agent.load(PICKLE_PATH)

    env.set_render_enabled(True)
    for ep in range(agent.episode, NUM_EPISODES):
        state = env.get_state()
        done = False
        total_reward = 0
        prev_bricks = 0
        prev_hits = 0

        while not done:
            action = agent.select_action(state)
            next_state, _, done = env.step(action)

            reward = calculate_reward(prev_bricks, prev_hits, env, done)
            total_reward += reward

            agent.update(state, action, reward, next_state)

            prev_bricks = env.get_bricks_broken_attempt()
            prev_hits = env.get_number_paddle_hits_round()
            state = next_state

        agent.remember_reward(total_reward)
        agent.episode += 1

        if (agent.episode) % LOG_INTERVAL == 0:
            agent.save(PICKLE_PATH)
            print(f"Episode {agent.episode} | Reward: {total_reward:.2f} | Logged to {PICKLE_PATH}")
        
        _, _, done = env.step(action)
        env.render()
        if done:
            env.reset()

    agent.save(PICKLE_PATH)


if __name__ == '__main__':
    train()