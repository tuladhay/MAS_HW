import numpy as np


class Agent:
    def __init__(self, n_nights):
        # Action is the night to choose to go to the bar
        self.n_actions = n_nights
        self.value_table = 100*np.random.uniform(self.n_actions)

    def choose_action(self):
        # choose the index of the max action in the value table
        pass

    def update_value_table(self, reward):
        # Here, the reward can be G,D or Local
        pass


class Elfabor:
    def __init__(self, n_agents, n_nights, b, alpha):
        # Total number of agents
        self.n_agents = n_agents
        # K nights that an agent can choose to attend the bar
        self.n_actions = n_nights
        # optimal number of attendees
        self.b = b
        # learning rate
        self.alpha = alpha

    def get_global_reward(self):
        pass

    def get_local_reward(self):
        pass

    def get_difference_reward(self):
        pass

    def reset(self):
        pass

    def step(self):
        pass
