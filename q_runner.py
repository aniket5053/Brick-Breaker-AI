from q_agent_breakout import QLearnerAgent

# To train
# agent = QLearnerAgent(render=False)
# agent.train(num_episodes=500000)

# # To play
agent = QLearnerAgent(start_level=6 , render=True)
agent.play()
