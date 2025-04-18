import torch
import numpy as np
import pickle
from breakout import Breakout
import os

ACTIONS = [0, 1, 2]  # 0: left, 1: right, 2: stay

class QAgent:
    def __init__(self):
        self.q_table = {}  # (state) -> torch.tensor of Q-values
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = torch.zeros(len(ACTIONS), dtype=torch.float32, device=self.device)
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS)
        qs = self.get_qs(state)
        return int(torch.argmax(qs).item())

    def learn(self, s, a, r, s_, done):
        q_s = self.get_qs(s)
        q_s_ = self.get_qs(s_)

        max_q_next = torch.max(q_s_) if not done else torch.tensor(0.0, device=self.device)
        target = r + self.gamma * max_q_next

        # Update Q-value using PyTorch tensor operations
        q_s[a] += self.alpha * (target - q_s[a])

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(n_episodes=500):
    env = Breakout()
    agent = QAgent()

    for ep in range(n_episodes):
        s = env.reset()
        total_reward = 0
        done = False

        while not done:
            env.render()
            a = agent.choose_action(s)
            s_, r, done = env.step(a)
            agent.learn(s, a, r, s_, done)
            s = s_
            total_reward += r

        print(f"Episode {ep+1}: Total reward = {total_reward}\t | Q-table size: {len(agent.q_table)}")

    env.close()

    # Save Q-table (convert tensors to list for pickling)
    q_table_serializable = {k: v.cpu().tolist() for k, v in agent.q_table.items()}
    if os.path.exists("q_table.pkl"):
        os.remove("q_table.pkl")
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table_serializable, f)

def play(n_episodes=5):
    with open("q_table.pkl", "rb") as f:
        q_table_loaded = pickle.load(f)

    env = Breakout()
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            env.render()
            q_values = q_table_loaded.get(s, [0.0, 0.0, 0.0])
            action = int(np.argmax(q_values))
            s, _, done = env.step(action)
        print(f"Completed Episode {ep+1}")
    env.close()

if __name__ == "__main__":
    train(2000)
    # play(5)
