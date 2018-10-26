import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class Agents:
    def __init__(self, n_agents, n_nights, b):
        self.n_agents = n_agents
        self.n_nights = n_nights
        self.b = b
        self.gamma = 0.1
        self.value_table = np.random.uniform(0, 1, [self.n_agents, self.n_nights])
        self.chosen_actions = []

    def select_actions(self):
        self.chosen_actions = np.argmax(self.value_table, 1)  # night that was chosen by each agent. size=n_agents
        # TODO: add epsilon-greedy

    def get_global_reward(self):
        # Calculate how many times each night was chosen by the agents
        nights_selected_dict = Counter(self.chosen_actions)
        global_reward = 0
        for k in range(self.n_nights):
            reward = nights_selected_dict[k]*np.exp(-nights_selected_dict[k]/self.b)
            global_reward += reward
        # return equally distributed reward
        return global_reward*np.ones(self.n_agents)/self.n_agents

    def get_local_reward(self):
        pass

    def get_difference_reward(self):
        pass

    def update_value_table(self, agent_reward):
        # agent_reward is a list with reward for each agent
        # the update has to depend on the action taken, and the agent_reward
        for agent in range(self.n_agents):
            self.value_table[agent, self.chosen_actions[agent]] += self.gamma*agent_reward[agent]

    def reset(self):
        pass


if __name__=="__main__":
    # Parameters
    n_agents = 5
    n_nights = 6
    b = 2
    n_episodes = 1000
    episode_returns = []

    for episode in range(n_episodes):
        agents = Agents(n_agents, n_nights, b)
        agents.select_actions()
        # Get reward for each agent as a list
        rewards = agents.get_global_reward()
        episode_returns.append(np.sum(rewards))  # this is the total rewards for all agents
        agents.update_value_table(rewards)

    plt.plot(episode_returns)
    plt.show()
