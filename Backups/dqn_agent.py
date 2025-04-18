import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pickle
import time
from collections import deque
from breakout import Breakout

# Global parameters
LR = 1e-3
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10
NUM_EPISODES = 500
STATE_SAVE_PATH = 'breakout_dqn.pkl'

# Simple network architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.action_dim = action_dim

        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        global EPSILON
        if random.random() < EPSILON:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions).squeeze()
        next_q = self.target_model(next_states).max(1)[0].detach()
        expected_q = rewards + (GAMMA * next_q * (1 - dones))

        loss = F.mse_loss(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Main training loop
def train():
    global EPSILON

    state_dim = 6
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(NUM_EPISODES):
        env = Breakout(log_to_csv=True)
        state = get_state(env)
        done = False
        total_reward = 0
        bricks_before_paddle_hit = 0

        while not done:
            action = agent.choose_action(state)

            # Execute action
            env.paddle.move_ip(-env.get_paddle_move_speed() if action == 0 else env.get_paddle_move_speed(), 0)
            env.paddle.clamp_ip((0, 0, 800, 600))

            next_state = get_state(env)
            done = env.get_isRoundLost()

            # Calculate reward according to specified conditions
            reward = 0
            if env.get_number_paddle_hits_round() > 0:
                reward += 1
                reward += 3 * bricks_before_paddle_hit
                bricks_before_paddle_hit = 0
            else:
                bricks_before_paddle_hit += env.get_bricks_broken_attempt()

            reward += 2 * env.get_bricks_broken_attempt()

            if done:
                reward -= 100

            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay()

        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        print(f'Episode {episode} Total Reward: {total_reward}, EPSILON: {EPSILON}')

        with open(STATE_SAVE_PATH, 'wb') as f:
            pickle.dump(agent.model.state_dict(), f)
    
    env.render()

# Helper functions
def get_state(env):
    return [
        env.get_ball_start_speed(),
        env.get_ball_acceleration_min(),
        env.get_ball_acceleration_max(),
        env.get_paddle_move_speed(),
        env.get_ball_hit_offset_multiplier(),
        env.get_paddle_velocity_multiplier()
    ]

if __name__ == '__main__':
    train()
