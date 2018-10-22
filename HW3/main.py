from environment import Agent, Elfabor

# number of agents
n_agents = 30
# optimal number of agents
b = 5
# number of nights
k = 7
# alpha value
alpha = 0.1

# create agents
agents = [Agent(k) for n in range(n_agents)]

# get the actions of all agents
joint_actions = [a.choose_action() for a in agents]

# Get rewards
