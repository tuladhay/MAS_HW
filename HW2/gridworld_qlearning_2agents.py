import numpy as np
import random
from entities import Agent, Target
import torch
import operator
import math
import matplotlib.pylab as plt


class Gridworld():
    def __init__(self, width, height, n_agents):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.agent1 = Agent()
        self.agent2 = Agent()
        self.target = Target()
        self.grid = np.zeros([self.width, self.height])

        self.set_agent_pos()
        self.set_target_pos()

        self.obs = []

    def set_agent_pos(self):
        pos_x = self.agent1.pos_x
        pos_y = self.agent1.pos_y
        pos_x2 = self.agent2.pos_x
        pos_y2 = self.agent2.pos_y
        self.grid[pos_x][pos_y] = 1
        self.grid[pos_x2][pos_y2] = 2

    def set_target_pos(self, random_pos=False):
        if random_pos:
            pos_x = random.randint(0, self.width)
            pos_y = random.randint(0, self.height)
            self.target.pos_x = pos_x  #check if this is correct
            self.target.pos_y = pos_y  #check if thi is correct
        else:
            pos_x = self.target.pos_x
            pos_y = self.target.pos_y
        self.grid[pos_x][pos_y] = 8

    def step(self, action, action2):
        done = False
        reward = 0
        reward_2 = 0
        reward -= 1  # negative reward for each step
        reward_2 -= 1
        # generate next action
        self.agent1.update_pos(action)
        self.agent2.update_pos(action2)
        self.target.move_random()  # TODO: remove this stationary target

        # update positions in the grid world
        self.grid_reset()
        self.set_agent_pos()
        self.set_target_pos(random_pos=False)

        # dist = math.sqrt((self.agent1.pos_x - self.target.pos_x) ** 2 + (self.agent1.pos_y - self.target.pos_y) ** 2)
        # reward = reward - abs(0.1*dist)

        # if agent 1 reaches the target
        if self.agent1.pos_x == self.target.pos_x and self.agent1.pos_y == self.target.pos_y:
            reward += 20
            done = True

        # if agent2 reaches the target
        if self.agent2.pos_x == self.target.pos_x and self.agent2.pos_y == self.target.pos_y:
            reward_2 += 20
            done = True

        # create observation vector of agent position(s) and target position
        observations = []
        observations.append(self.agent1.pos_x)  # agent1
        observations.append(self.agent1.pos_y)
        observations.append(self.agent2.pos_x)  # agent2
        observations.append(self.agent2.pos_y)
        observations.append(self.target.pos_x)
        observations.append(self.target.pos_y)

        return observations, reward, reward_2, done

    def reset(self):
        self.agent1.reset()
        self.agent2.reset()
        self.target.reset()

        observations = []
        observations.append(self.agent1.pos_x)  # agent1
        observations.append(self.agent1.pos_y)
        observations.append(self.agent2.pos_x)  # agent2
        observations.append(self.agent2.pos_y)
        observations.append(self.target.pos_x)
        observations.append(self.target.pos_y)
        return observations

    def grid_reset(self):
        self.grid = np.zeros([self.width, self.height])


if __name__ == "__main__":
    ### Args ###
    num_steps = 200
    num_episodes = 20000
    num_actions = 4

    gamma = 0.8
    lamda = 0.25
    epsilon = 0.1  # e-greedy

    ######################

    game = Gridworld(10, 5, 1)
    episode_reward_list = []    # Agent1 iterate this is the one to plot per epoch
    episode_reward_list_2 = []  # Agent2

    # make a Q-table dictionary
    q_table = {}
    q_table_2 = {}

    for i in range(10):  # agent pos x
        for j in range(5):  # agent pos y
            for i2 in range(10):  # agent2 pos x
                for j2 in range(5):  # agent2 pos y
                    for k in range(10):  # target pos x
                        for l in range(5):  # target pos y
                            for a in range(4):  # actions
                                q_table[(i, j, i2, j2, k, l, a)] = 10
                                q_table_2[(i, j, i2, j2, k, l, a)] = 10
    print("Q-table initialized")

    for i_episode in range(num_episodes):
        obs = game.reset()  #reset at start of each episode
        episode_reward = 0  # agent1
        episode_reward_2 = 0  # agent2
        for t in range(num_steps):  #TODO: make args
            ''' get actions for that state '''
            action_values = []
            action_values_2 = []

            # for agent 1
            for n in range(num_actions):
                action_values.append(q_table[(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], n)])

            # for agent 2
            for n in range(num_actions):
                action_values_2.append(q_table_2[(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], n)])

            # get max value for Agent1
            if all(action_values[0] == v for v in action_values):  # when all values are equal
                action = random.randint(0, num_actions-1)
                current_act_value = action_values[0]
            else:
                index, current_act_value = max(enumerate(action_values), key=operator.itemgetter(1))
                action = index

            # get max value for Agent2
            if all(action_values_2[0] == v for v in action_values_2):  # when all values are equal
                action_2 = random.randint(0, num_actions-1)
                current_act_value_2 = action_values_2[0]
            else:
                index2, current_act_value_2 = max(enumerate(action_values_2), key=operator.itemgetter(1))
                action_2 = index2

            # epsilon greedy agent1
            random_num = random.random()
            if random_num < epsilon:
                action = random.randint(0, 3)
            # epsilon greedy agent2
            if random_num < (1-epsilon):
                action_2 = random.randint(0, 3)

            # take one step
            next_obs, reward, reward_2, done = game.step(action, action_2)
            episode_reward += reward
            episode_reward_2 += reward_2

            ''' get value for next state Q(s', a') '''
            action_values = []
            action_values_2 = []

            for n in range(num_actions):
                action_values.append(q_table[(next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5], n)])
                action_values_2.append(q_table_2[(next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5], n)])

            # get max value
            _, next_act_value = max(enumerate(action_values), key=operator.itemgetter(1))  # we have the value for Q(s',a') now
            _, next_act_value_2 = max(enumerate(action_values_2), key=operator.itemgetter(1))

            # update the q-values
            q_table[(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], action)] += lamda*(reward + gamma*next_act_value - current_act_value)
            q_table_2[(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], action_2)] += lamda * (reward_2 + gamma * next_act_value_2 - current_act_value_2)

            obs = next_obs
            if done:
                break

        if i_episode%100==0:
            print("Episode " + str(i_episode) + ". Total reward: " + str(episode_reward))
        episode_reward_list.append(episode_reward)
        episode_reward_list_2.append(episode_reward_2)

    plt.plot(episode_reward_list)
    plt.plot(episode_reward_list_2)
    plt.title("Total rewards vs episodes while learning")
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.show()

    ''' Now for testing '''
    test_episode_reward_list = []
    test_episode_reward_list_2 = []
    for i_episode in range(num_episodes):
        obs = game.reset()  #reset at start of each episode
        episode_reward = 0
        episode_reward_2 = 0
        for t in range(num_steps):
            ''' get actions for that state '''
            action_values = []
            action_values_2 = []
            for n in range(num_actions):
                action_values.append(q_table[(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], n)])
                action_values_2.append(q_table_2[(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], n)])

            # get max valuefor Agent1
            if all(action_values[0] == v for v in action_values):  # when all values are equal
                action = random.randint(0, num_actions-1)
                current_act_value = action_values[0]
            else:
                index, current_act_value = max(enumerate(action_values), key=operator.itemgetter(1))
                action = index

            # get max value for Agent2
            if all(action_values_2[0] == v for v in action_values_2):  # when all values are equal
                action_2 = random.randint(0, num_actions-1)
                current_act_value_2 = action_values_2[0]
            else:
                index2, current_act_value_2 = max(enumerate(action_values_2), key=operator.itemgetter(1))
                action_2 = index2

            # take one step
            next_obs, reward, reward_2, done = game.step(action, action_2)
            # accumulate rewards
            episode_reward += reward
            episode_reward_2 += reward_2

            obs = next_obs
            # print(game.grid)
            if done:
                break

        if i_episode % 100 == 0:
            print("Test Episode " + str(i_episode) + ". Total reward: " + str(episode_reward))
        test_episode_reward_list.append(episode_reward)
        test_episode_reward_list_2.append(episode_reward_2)

    plt.plot(test_episode_reward_list)
    plt.plot(test_episode_reward_list_2)
    plt.title("Total rewards vs episodes while testing")
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.ylim(-250, 25)
    plt.show()
