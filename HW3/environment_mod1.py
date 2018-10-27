import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
from copy import copy

# Mod: changing the rewards from equally divided, to straighforward G reward for each agent

class Agents:
    def __init__(self, n_agents, n_nights, b, epsilon):
        self.n_agents = n_agents
        self.n_nights = n_nights
        self.b = b
        self.gamma = 0.1  # learning rate
        np.random.seed(0)
        self.value_table = np.random.uniform(0, 10, [self.n_agents, self.n_nights])
        self.chosen_actions = []
        self.cf_actions = []
        self.epsilon = epsilon
        self.global_rewards = []  # carries a list of reward for each agent. Sum it to get G

        self.temp = []

    def select_actions(self):
        self.chosen_actions = np.argmax(self.value_table, 1)  # night that was chosen by each agent. list size=n_agents
        # epsilon-greedy
        for a in range(len(self.chosen_actions)):
            if random.random()<self.epsilon:
                self.chosen_actions[a] = random.randint(0, self.n_nights-1)

    def get_global_reward(self, counterfactual=False):
        # Calculate how many times each night was chosen by the agents
        nights_selected_dict = Counter(self.chosen_actions)
        if counterfactual:
            nights_selected_dict = Counter(self.cf_actions)
        global_reward = 0
        for k in range(self.n_nights):
            reward = nights_selected_dict[k]*np.exp(-nights_selected_dict[k]/self.b)
            global_reward += reward
        # return equally distributed reward
        self.global_rewards = global_reward*np.ones(self.n_agents)
        return self.global_rewards  # This is a list for individual agent rewards. Sum to get G

    def get_local_reward(self):
        pass

    def get_difference_reward(self):
        G_reward = self.get_global_reward()
        d_reward_list = np.zeros(self.n_agents)
        for a in range(self.n_agents):
            random_action = random.randint(0, n_nights-1)
            # copy what the actual actions were
            self.cf_actions = copy(self.chosen_actions)
            # replace with a counterfactual action
            self.cf_actions[a] = random_action
            # calculate the difference reward for this agent
            cf = np.sum(self.get_global_reward(counterfactual=True))
            #agent_d_reward = np.sum(self.get_global_reward()) - np.sum(self.get_global_reward(counterfactual=True))
            agent_d_reward = self.get_global_reward()[a] - self.get_global_reward(counterfactual=True)[a]
            # append it to list for each agent difference reward
            d_reward_list[a] = agent_d_reward

        #self.temp.append(np.sum(self.get_global_reward(counterfactual=True)))
        # save G that was trained using Difference Rewards -> Connor
        self.temp.append(np.sum(G_reward))
            #print(np.sum(self.get_global_reward(counterfactual=True)[a]))
            #print(str(self.get_global_reward()[a]) + " - " + str(self.get_global_reward(counterfactual=True)[a] )  + " = " + str(agent_d_reward) )
        return d_reward_list

    def update_value_table(self, agent_reward):
        # agent_reward is a list with reward for each agent
        # the update has to depend on the action taken, and the agent_reward
        for agent in range(self.n_agents):
            self.value_table[agent, self.chosen_actions[agent]] += self.gamma*(agent_reward[agent]-self.value_table[agent, self.chosen_actions[agent]])


if __name__=="__main__":
    # Parameters
    n_agents = 30
    n_nights = 7
    b = 5
    n_episodes = 5000
    epsilon = 0.05
    episode_returns = []
    agents = Agents(n_agents, n_nights, b, epsilon)

    for episode in range(n_episodes):
        agents.select_actions()
        # Get reward for each agent as a list
        rewards = agents.get_global_reward()
        #rewards = agents.get_difference_reward()
        #print(rewards)
        episode_returns.append(np.sum(rewards))  # this is the total rewards for all agents
        agents.update_value_table(rewards)

    plt.plot(episode_returns)
    #plt.show()

    agents = Agents(n_agents, n_nights, b, epsilon)
    episode_returns_d = []
    for episode in range(n_episodes):
        agents.select_actions()
        # Get reward for each agent as a list
        #rewards = agents.get_global_reward()
        rewards = agents.get_difference_reward()
        episode_returns_d.append(np.sum(rewards))  # this is the total rewards for all agents
        agents.update_value_table(rewards)
    #plt.plot(episode_returns_d)
    plt.plot(agents.temp)
    plt.show()
