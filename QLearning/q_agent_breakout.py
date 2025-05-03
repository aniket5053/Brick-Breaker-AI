import pygame # Only to handle window close event here
import numpy as np
import pickle
import gzip
import signal
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from breakout import Breakout

QTABLE_FILE = "q_learning_table.pkl.gz"
REWARD_FILE = "q_learning_reward_history.pkl.gz"
QLEARN_PLOT = "QLearning_PerformancePlot.png"

# locked_level=<> add this to breakout init to lock the current level
# Extend to QLearnerAgent to lock the agent to a level
class QLearnerAgent:
    def __init__(self, start_level=0, locked_level=None, render=False):
        self.start_level = start_level
        self.locked_level = locked_level
        self.env = Breakout(start_level=self.start_level, locked_level=self.locked_level, log_to_csv=True)
        self.env.set_render_enabled(render)

        self.q_table = self._load_q_table()
        self.rewards = self._load_rewards()
        self.episodes = len(self.rewards)
        self.playing = False

        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.session_rewards = []
        self.session_start_episode = self.episodes  # Mark start of this session

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *_):
        if not self.playing:
            self._save_q_table()
            print("    Q-table saved on exit!")
            self._save_rewards()
            print("    Reward history saved!")
            self._plot_rewards()
            print("    Plot saved on exit!")
            self.env.log_and_quit()
        else:
            print("Stoping game play...")
            pygame.quit()
            sys.exit()

    def _load_q_table(self):
        if os.path.exists(os.path.join("output", QTABLE_FILE)):
            with gzip.open(os.path.join("output", QTABLE_FILE), "rb") as f:
                return pickle.load(f)
        return {}

    def _save_q_table(self):
        os.makedirs("output", exist_ok=True)
        with gzip.open(os.path.join("output", QTABLE_FILE), "wb") as f:
            pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved Q-table with {len(self.q_table)} entries (compressed).")

    def _load_rewards(self):
        if os.path.exists(os.path.join("output", REWARD_FILE)):
            with gzip.open(os.path.join("output", REWARD_FILE), "rb") as f:
                return pickle.load(f)
        return []

    def _save_rewards(self):
        os.makedirs("output", exist_ok=True)
        with gzip.open(os.path.join("output", REWARD_FILE), "wb") as f:
            pickle.dump(self.rewards, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        return self.q_table[state]

    def _plot_rewards(self):
        if not self.rewards:
            return

        episode_range = range(1, len(self.rewards) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(episode_range, self.rewards, label="Total Reward per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Q-Learning Reward v. Episodes @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        os.makedirs("output", exist_ok=True)
        plt.savefig(os.path.join("output", QLEARN_PLOT))
        plt.close()
        print(f"      Reward plot saved to './output/{QLEARN_PLOT}'")

    def train(self, num_episodes=5000):
        self.session_rewards = []
        self.session_start_episode = self.episodes

        for _ in range(num_episodes):
            self.episodes += 1
            self.env.reset()
            state = self.env.get_state()
            total_reward = 0
            prev_bricks = 0
            prev_hits = 0
            prev_level = self.env.get_current_level()
            done = False

            while not done:
                if self.env.render_enabled:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self._handle_exit()

                action = np.random.choice(3) if np.random.rand() < self.epsilon else np.argmax(self._get_qs(state))
                next_state, _, done = self.env.step(action)

                reward = 0
                bricks = self.env.get_bricks_broken_attempt()
                hits = self.env.get_number_paddle_hits_round()
                level = self.env.get_current_level()
                attempt_time = self.env.get_attempt_time()

                # Rewards
                if self.env.get_isRoundLost():
                    reward -= 100  # Failed the current attempt
                elif not done and bricks > prev_bricks:
                    reward += 2 + (3 * (bricks - prev_bricks))  # Hit breakable brick before hitting paddle
                elif not done and hits > prev_hits:
                    reward += 0.5  # Hit the paddle (still alive)
                
                if done and not self.env.get_isRoundLost():
                    reward += 100  # Beat the Level
                if level > prev_level and level % 7 == 0:
                    reward += 250  # Passed level 7, one time bonus
                if done and hits > 10 and bricks == prev_bricks:
                    reward -= 20  # Penalize too many bounces without progress
                if done and attempt_time < 30 and not self.env.get_isRoundLost() and level != prev_level:
                    reward += 50  # Reward for quickly clearing, within x seconds
                if done and attempt_time > 120 and not self.env.get_isRoundLost():
                    reward -= 200 # If trapped on a round

                reward += 0.05  # Bonus for still being alive

                total_reward += reward
                prev_bricks = bricks
                prev_hits = hits
                prev_level = level

                old_q = self._get_qs(state)[action]
                next_max_q = np.max(self._get_qs(next_state))
                self.q_table[state][action] = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)

                state = next_state
                self.env.render()

            self.rewards.append(total_reward)
            self.session_rewards.append(total_reward)

            print(f"Episode {self.episodes} | Reward: {total_reward:.1f} | Q-table size: {len(self.q_table)} | ε = {self.epsilon} | Level: {level+1} | Attempts: {self.env.get_total_attempts_round()} | Broke [{bricks}] bricks")

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if self.episodes % 500 == 0:
                self._save_q_table()
                print(f"    Q-Table Saved on Episode {self.episodes}!")
                self._save_rewards()
                print(f"    Rewards Saved on Episode {self.episodes}!")
                self._plot_rewards()
                print(f"    Plot Saved on Episode {self.episodes}!")
                self.env._log_csv(f"training_run_ep {self.episodes}")

        self._save_q_table()
        self._save_rewards()
        self._plot_rewards()
        self.env._log_csv(f"training_end_ep {self.episodes}")
        print("Training Complete!")

    def play(self):
        self.env.set_render_enabled(True)
        self.playing = True
        while True:
            if self.env.render_enabled:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._handle_exit()
            self.env.reset()
            state = self.env.get_state()
            done = False
            while not done:
                action = np.argmax(self._get_qs(state))
                state, _, done = self.env.step(action)
                self.env.render()
