# q_learning_agent.py
import numpy as np
import pickle
from breakout import Breakout
import os

ACTIONS = [0, 1, 2]  # 0: left, 1: right, 2: stay

class QAgent:
    def __init__(self):
        self.q_table = {}  # (state) -> [Q values for actions]
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in ACTIONS]
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS)
        return int(np.argmax(self.get_qs(state)))

    def learn(self, s, a, r, s_, done):
        if s not in self.q_table:
            self.q_table[s] = [0.0 for _ in ACTIONS]
        if s_ not in self.q_table:
            self.q_table[s_] = [0.0 for _ in ACTIONS]

        max_q_next = max(self.q_table[s_]) if not done else 0
        target = r + self.gamma * max_q_next
        self.q_table[s][a] += self.alpha * (target - self.q_table[s][a])

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

    if os.path.exists("q_table.pkl"):
        os.remove("q_table.pkl")
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

def play(n_episodes=5):
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    env = Breakout()
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            env.render()
            action = np.argmax(q_table.get(s, [0, 0, 0]))  # default to zeros if state is missing
            s, _, done = env.step(action)
        print(f"Completed Episode {ep+1}")
    env.close()

if __name__ == "__main__":
    train(2000)
    # play(2000)
