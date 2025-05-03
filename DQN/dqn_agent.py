"""
DQN-based Breakout Training and Evaluation Script.

This script trains a Deep Q-Network (DQN) agent to play a Breakout game using
PyTorch and a custom `Breakout` environment. It supports both training and playing
modes and saves progress in checkpoints and plots.

Main components:
- PrioritizedReplayBuffer: Experience replay with prioritization.
- DQN: Neural network approximating Q-values.
- DQNAgent: Handles action selection, training, and model persistence.
- Helper functions: Reward calculation, plotting, game state extraction.
- Main training loop and interactive play loop.
authoer: Antar Chowdhury
"""

import os
import random
import signal
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from breakout import Breakout

# Define directories and device
STATE_SAVE_DIR = 'checkpoints'
PLOT_SAVE_DIR = 'plots'
FINAL_MODEL_PATH = os.path.join(STATE_SAVE_DIR, 'finalmodel_02052025.pth')
LATEST_CHECKPOINT = os.path.join(STATE_SAVE_DIR, 'finalmodel_02052025.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(STATE_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Hyperparameters
LR = 1e-4
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01
MEMORY_SIZE = 50000
BATCH_SIZE = 64
TARGET_UPDATE = 10
NUM_EPISODES = 10000

class PrioritizedReplayBuffer:
    """Experience replay buffer with prioritized sampling."""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, transition, td_error=1.0):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return [], []
        scaled_priorities = self.priorities[:self.size]
        prob = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=prob)
        transitions = [self.buffer[i] for i in indices]
        return transitions, indices

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = (abs(td_error) + 1e-5) ** self.alpha

class DQN(nn.Module):
    """Simple feedforward neural network for Q-value estimation."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Deep Q-Network agent with prioritized experience replay."""
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.update_target()

    def load_weights(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=device))
            self.update_target()
            print(f"Loaded model weights from {path}")

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        global EPSILON
        if random.random() < EPSILON:
            return random.choice([0, 1, 2])
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def replay(self):
        if self.memory.size < BATCH_SIZE:
            return
        batch, indices = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q = self.model(states).gather(1, actions).squeeze()
        next_q = self.target_model(next_states).max(1)[0].detach()
        expected_q = rewards + GAMMA * next_q * (1 - dones)

        td_errors = expected_q - current_q
        loss = F.mse_loss(current_q, expected_q)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.scheduler.step()

def get_state(game):
    """Extracts and normalizes the current state from the game environment."""
    ball_x, ball_y = game.get_ball_position()
    paddle_x = game.get_paddle_position()
    screen_width = game.get_screen_width()
    screen_height = game.get_screen_height()

    return [
        ball_x / screen_width,
        ball_y / screen_height,
        paddle_x / screen_width,
        0 if game.ball_dx < 0 else 1,
        0 if game.ball_dy < 0 else 1
    ]

def calculate_reward(prev_bricks, prev_hits, game, done):
    """Calculates a shaped reward based on game events and performance."""
    reward = 0
    bricks = game.get_bricks_broken_attempt()
    hits = game.get_number_paddle_hits_round()
    attempt_time = game.get_attempt_time()

    try:
        paddle_x = game.get_paddle_position()
        ball_x = game.get_ball_position()[0]
        screen_width = game.get_screen_width()
        norm_distance = abs(paddle_x - ball_x) / screen_width
        reward -= 0.2 * norm_distance
    except:
        pass

    if not done:
        reward += 0.1

    if game.get_isRoundLost():
        # print(reward)
        reward -= 1000
        # print("Round lost!")
    elif not done and bricks > prev_bricks:
        reward += 5 * (bricks - prev_bricks)
    elif not done and hits > prev_hits:
        reward += 1

    if done and not game.get_isRoundLost():
        reward += 100
        if attempt_time < 30:
            reward += 50

    if done and hits > 10 and bricks == prev_bricks:
        reward -= 20
    if not done and attempt_time > 200:
        reward -= 100
    reward += 0.2
    return reward

def plot_rewards(reward_list):
    """Plots and saves the reward progress over episodes."""
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list)
    curr_time = datetime.now()
    format_time = curr_time.strftime("%Y-%m-%d_%H-%M-%S")
    plot_path = os.path.join(PLOT_SAVE_DIR, f'reward_plot_{format_time}.png')
    plt.title(f"Reward Progress over Episodes at {format_time}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

def plot_bricks(bricks_list):
    """Plots and saves the number of bricks broken per episode."""
    plt.figure(figsize=(10, 5))
    plt.plot(bricks_list)
    curr_time = datetime.now()
    format_time = curr_time.strftime("%Y-%m-%d_%H-%M-%S")
    plot_path = os.path.join(PLOT_SAVE_DIR, f'bricks_plot_{format_time}.png')
    plt.title(f"Bricks Broken per Episode at {format_time}")
    plt.xlabel("Episode")
    plt.ylabel("Bricks Broken")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

def train():
    """Main training loop for the DQN agent."""
    global EPSILON

    env = Breakout(log_to_csv=True)
    state_dim = 5
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim)

    agent.load_weights(LATEST_CHECKPOINT)

    reward_log = []
    bricks_log = []
    interrupted = False

    def handle_interrupt(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nTraining interrupted. Saving model and plots...")

    signal.signal(signal.SIGINT, handle_interrupt)

    for episode in range(NUM_EPISODES):
        if interrupted:
            break

        env.reset()
        state = get_state(env)
        done = False
        total_reward = 0
        prev_bricks = 0
        prev_hits = 0

        env.set_render_enabled(False)

        while not done:
            action = agent.act(state)
            next_state, _, done = env.step(action)

            reward = calculate_reward(prev_bricks, prev_hits, env, done)
            # print(f"Calculated Reward: {reward:.2f}")
            prev_bricks = env.get_bricks_broken_attempt()
            prev_hits = env.get_number_paddle_hits_round()

            reward = np.clip(reward, -1000, 100000)
            # print(f"Reward after clipping: {reward:.2f}")
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # print(f"Total Reward: {total_reward:.2f}")

            agent.replay()

        if episode % 500 == 0 and EPSILON < 0.2:
            EPSILON = 0.5
        else:
            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        reward_log.append(total_reward)
        bricks_broken = env.get_bricks_broken_attempt()
        bricks_log.append(bricks_broken)

        if (episode + 1) % 100 == 0:
            plot_rewards(reward_log)
            plot_bricks(bricks_log)
            checkpoint_file = os.path.join(STATE_SAVE_DIR, f'checkpoint_ep{episode+1:03d}.pth')
            agent.save_weights(checkpoint_file)
            agent.save_weights(LATEST_CHECKPOINT)

        if (episode + 1) % TARGET_UPDATE == 0:
            agent.update_target()

        print(f"Episode {episode+1:03d} | Reward = {total_reward:.2f} | Buffer Size = {agent.memory.size} | ε = {EPSILON:.3f} | Bricks = {env.get_bricks_broken_attempt()}")

    agent.save_weights(FINAL_MODEL_PATH)
    agent.save_weights(LATEST_CHECKPOINT)
    print("Training completed and saved.")

def play():
    """Runs the game using the trained model for demonstration."""
    env = Breakout(log_to_csv=True)
    state_dim = 5
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim)

    agent.load_weights(LATEST_CHECKPOINT)

    env.set_render_enabled(True)

    while True:
        env.reset()
        state = get_state(env)
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.act(state)
            next_state, _, done = env.step(action)
            state = next_state
            total_reward += 0  # Optionally accumulate rewards here
            time.sleep(0.01)

        print(f"Game Over. Total Reward: {total_reward}")

    env.close()

if __name__ == '__main__':
    mode = input("Enter mode (train/play): ").strip().lower()
    if mode == "train":
        train()
    elif mode == "play":
        play()
    else:
        print("Invalid input. Please enter 'train' or 'play'.")

