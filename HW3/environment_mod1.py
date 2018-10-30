import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
from copy import copy
import seaborn as sns; sns.set(color_codes=True)


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

        self.temp = []  # G that was trained using difference rewards
        self.temp2 = []
        self.local = []  # G that was trained using local rewards

    def select_actions(self):
        self.chosen_actions = np.argmax(self.value_table, 1)  # night that was chosen by each agent. list size=n_agents
        # epsilon-greedy
        for a in range(len(self.chosen_actions)):
            if random.random()<self.epsilon:
                self.chosen_actions[a] = random.randint(0, self.n_nights-1)

    def get_global_reward(self, counterfactual=False):
        # For global rewards we don't have to loop over agents since it's "Global"
        # Calculate how many times each night was chosen by the agents
        nights_selected_dict = Counter(self.chosen_actions)
        if counterfactual:
            nights_selected_dict = Counter(self.cf_actions)
        global_reward = 0
        for k in range(self.n_nights):
            reward = nights_selected_dict[k]*np.exp(-nights_selected_dict[k]/self.b)
            global_reward += reward
        # same reward for all agents
        self.global_rewards = global_reward*np.ones(self.n_agents)
        return self.global_rewards  # This is a list for individual agent rewards. Sum to get G

    def get_local_reward(self):
        # Reward for the night that the agent chose to visit the bar
        # G that was trained using local rewards
        G_reward = self.get_global_reward()
        local_reward = 0
        local_reward_list = np.zeros(self.n_agents)
        nights_selected_dict = Counter(self.chosen_actions)
        # for each agent, reward = agents that showed up the same night * exp(-agents that showed up same night / b)
        for a in range(self.n_agents):
            local_reward = nights_selected_dict[self.chosen_actions[a]]*np.exp(-nights_selected_dict[self.chosen_actions[a]]/self.b)
            local_reward_list[a] = local_reward
        self.local.append(np.sum(G_reward))
        return local_reward_list

    def get_difference_reward(self):
        '''Counterfactual = Random action'''
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

        # save G that was trained using Difference Rewards -> Connor
        self.temp.append(np.sum(G_reward))
        return d_reward_list

    def get_difference_reward_2(self):
        '''Counterfactual = self removal'''
        G_reward = self.get_global_reward()
        d2_reward_list = np.zeros(self.n_agents)

        for a in range(self.n_agents):
            # copy what the actual actions were
            self.cf_actions = copy(self.chosen_actions)
            # replace with a counterfactual action
            self.cf_actions[a] = 100
            '''The idea here is that if the cf_action is set to 100, then in the get_global_reward(counterfactual=True),
            since it loops over n_night and finds the number of agents in dictionary corresponding to the night, it will
            never get to night=100, since the loop is over n_nights. So basically the agent would not be counted in the 
            reward calculation'''
            # calculate the difference reward for this agent
            agent_d_reward = self.get_global_reward()[a] - self.get_global_reward(counterfactual=True)[a]
            # append it to list for each agent difference reward
            d2_reward_list[a] = agent_d_reward

        # save G that was trained using Difference Rewards -> Connor
        self.temp2.append(np.sum(G_reward))
        return d2_reward_list


    def update_value_table(self, agent_reward):
        # agent_reward is a list with reward for each agent
        # the update has to depend on the action taken, and the agent_reward
        for agent in range(self.n_agents):
            self.value_table[agent, self.chosen_actions[agent]] += self.gamma*(agent_reward[agent]-self.value_table[agent, self.chosen_actions[agent]])


#if __name__=="__main__":
def main():
    '''Global Rewards'''
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
    #plt.plot(np.divide(episode_returns,n_agents))
    plt.xlabel("Episodes (Weeks)")
    plt.ylabel("Total Returns")
    plt.title("Global Reward, Agents=30, k=7, b=5")
    #print(agents.chosen_actions)
    h = np.hstack(agents.chosen_actions)

    # for statistical runs
    stat_g_returns.append(episode_returns)

    '''Difference Rewards'''
    agents = Agents(n_agents, n_nights, b, epsilon)
    for episode in range(n_episodes):
        agents.select_actions()
        # Get difference reward for each agent as a list
        rewards = agents.get_difference_reward()
        agents.update_value_table(rewards)
    #plt.plot(np.divide(agents.temp,n_agents))
    h_d = np.hstack(agents.chosen_actions)
    plt.xlabel("Episodes (Weeks)")
    plt.ylabel("Total Returns")
    plt.title("Difference Reward (random action), Agents=30, k=7, b=5")

    # for statistical runs
    stat_d_returns.append(agents.temp)


    '''Difference Rewards Removed Agent'''
    agents = Agents(n_agents, n_nights, b, epsilon)
    for episode in range(n_episodes):
        agents.select_actions()
        # Get difference reward for each agent as a list
        rewards = agents.get_difference_reward_2()
        agents.update_value_table(rewards)

    h_d2 = np.hstack(agents.chosen_actions)
    #plt.plot(np.divide(agents.temp2,n_agents))
    plt.xlabel("Episodes (Weeks)")
    plt.ylabel("Total Returns")
    plt.title("Difference Reward (agent removed), Agents=30, k=7, b=5")

    # for statistical runs
    stat_d2_returns.append(agents.temp2)


    '''Local Rewards'''
    agents = Agents(n_agents, n_nights, b, epsilon)
    local_episode_returns = []
    for episode in range(n_episodes):
        agents.select_actions()
        # get local reward for each agent as a list
        rewards = agents.get_local_reward()
        agents.update_value_table(rewards)
    h_l = np.hstack(agents.chosen_actions)
    #plt.plot(np.divide(agents.local,n_agents))
    plt.xlabel("Episodes (Weeks)")
    plt.ylabel("Total Returns")
    plt.title("Local Reward, Agents=30, k=7, b=5")

    # for statistical runs
    stat_l_returns.append(agents.local)

    # show rewards plot
    #plt.show()

    # plot histograms
    # plt.hist(h, bins=[0,1,2,3,4,5,6])
    # plt.title("Global")
    # plt.xlabel("nights")
    # plt.ylabel("Agents Present")
    # plt.show()
    #
    # plt.hist(h_d, bins=[0,1,2,3,4,5,6])
    # plt.title("Difference Rewards (Random Action")
    # plt.xlabel("nights")
    # plt.ylabel("Agents Present")
    # plt.show()
    #
    # plt.hist(h_d2, bins=[0,1,2,3,4,5,6])
    # plt.title("Difference Rewards (Agent Removed)")
    # plt.xlabel("nights")
    # plt.ylabel("Agents Present")
    # plt.show()
    #
    # plt.hist(h_l, bins=[0,1,2,3,4,5,6,7])
    # plt.title("Local Rewards")
    # plt.xlabel("nights")
    # plt.ylabel("Agents Present")
    # plt.show()




if __name__=="__main__":
    # Parameters
    n_agents = 40
    n_nights = 5
    b = 4
    n_episodes = 1500
    epsilon = 0.1

    stat_runs = 10

    stat_g_returns = []
    stat_d_returns = []
    stat_d2_returns = []
    stat_l_returns = []


    for n in range(stat_runs):
        main()

    print()
    x = [n for n in range(0, n_episodes, 20)]
    g = []
    for n in range(stat_runs):
        g.append(stat_g_returns[n][0::20])
    g_mean = np.mean(g, axis=0)
    std = np.std(g, axis=0) / np.sqrt(stat_runs)
    plt.errorbar(x, np.divide(g_mean, n_agents), yerr=np.divide(std,n_agents))


    d = []
    for n in range(stat_runs):
        d.append(stat_d_returns[n][0::20])
    d_mean = np.mean(d, axis=0)
    std = np.std(d, axis=0) / np.sqrt(stat_runs)
    plt.errorbar(x, np.divide(d_mean, n_agents), yerr=np.divide(std,n_agents))

    d2 = []
    for n in range(stat_runs):
        d2.append(stat_d2_returns[n][0::20])
    d2_mean = np.mean(d2, axis=0)
    std = np.std(d2, axis=0) / np.sqrt(stat_runs)
    plt.errorbar(x, np.divide(d2_mean, n_agents), yerr=np.divide(std,n_agents))


    l = []
    for n in range(stat_runs):
        l.append(stat_l_returns[n][0::20])
    l_mean = np.mean(l, axis=0)
    std = np.std(l, axis=0) / np.sqrt(stat_runs)
    plt.errorbar(x, np.divide(l_mean, n_agents), yerr=np.divide(std,n_agents))

    plt.title("Agents:40, K=5, b=4")
    plt.show()
