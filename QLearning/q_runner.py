from q_agent_breakout import QLearnerAgent

# To train
# agent = QLearnerAgent(start_level=0, locked_level=0, render=False)
# agent.train(num_episodes=100000)

# # To play
agent = QLearnerAgent(start_level=0, locked_level=0, render=True)
agent.play()
