import numpy as np
import random
from operator import attrgetter
from entities import Agent, Target
import torch
import operator
from replay_memory import ReplayMemory, Transition
import math


class Gridworld():
    def __init__(self, width, height, n_agents):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.agent1 = Agent()
        self.target = Target()
        self.grid = np.zeros([self.width, self.height])
        self.timestep = 0

        self.set_agent_pos()
        self.set_target_pos()

        self.obs = []

    def set_agent_pos(self):
        pos_x = self.agent1.pos_x
        pos_y = self.agent1.pos_y
        self.grid[pos_x][pos_y] = 1

    def set_target_pos(self, random_pos=False):
        if random_pos:
            pos_x = random.randint(0, self.width)
            pos_y = random.randint(0, self.height)
        else:
            pos_x = self.target.pos_x
            pos_y = self.target.pos_y
        self.grid[pos_x][pos_y] = 8

    def step(self, action):
        done = False
        reward = 0
        reward -= 1  # negative reward for each step
        # generate next action
        self.agent1.update_pos(action)

        #self.target.move_random()  # TODO: remove this stationary target

        # update positions in the grid world
        self.grid_reset()
        self.set_agent_pos()
        self.set_target_pos(random_pos=False)

        #dist = math.sqrt((self.agents[0].pos_x - self.target.pos_x) ** 2 + (self.agents[0].pos_y - self.target.pos_y) ** 2)
        #reward = reward - abs(0.5*dist)

        if self.agent1.pos_x == self.target.pos_x and self.agent1.pos_y == self.target.pos_y:
            reward += 100
            done = True

        self.timestep += 1
        #print("reward : "+str(reward)+str("\n"))

        # create observation vector of agent position(s) and target position
        observations = []
        observations.append(self.agent1.pos_x)
        observations.append(self.agent1.pos_y)
        observations.append(self.target.pos_x)
        observations.append(self.target.pos_y)

        return observations, reward, done

    def reset(self):
        self.timestep = 0
        self.agent1.reset()
        self.target.reset()

        observations = []
        observations.append(self.agent1.pos_x)
        observations.append(self.agent1.pos_y)
        observations.append(self.target.pos_x)
        observations.append(self.target.pos_y)
        return observations

    def grid_reset(self):
        self.grid = np.zeros([self.width, self.height])


if __name__ == "__main__":
    ### Args ###
    num_steps = 50
    batch_size = 16
    updates_per_step = 5
    num_episodes = 1000
    num_actions = 4

    gamma = 0.99
    lamda = 0.1

    ######################

    game = Gridworld(10, 5, 1)
    episode_reward_list = []  # iterate this is the one to plot per epoch

    ### make a Q-table dictionary
    q_table = {}

    for i in range(10):  #agent pos x
        for j in range(5):  # agent pos y
            for k in range(10):  # target pos x
                for l in range(5): # target pos y
                    for a in range(4):  # actions
                        q_table[(i,j,k,l,a)] = 100

    print("Q-table initialized")

    for i_episode in range(num_episodes):
        obs = game.reset()  #reset at start of each episode
        episode_reward = 0
        for t in range(num_steps):  #TODO: make args
            ''' get actions for that state '''
            action_values = []
            for n in range(num_actions):
                action_values.append(q_table[(obs[0], obs[1], obs[2], obs[3], n)])  # FOR SINGE AGENT and TARGET

            # get max value
            if all(action_values[0] == v for v in action_values):  # when all values are equal
                action = random.randint(0, num_actions-1)
                current_act_value = action_values[0]
            else:
                index, current_act_value = max(enumerate(action_values), key=operator.itemgetter(1))
                action = index

            # take one step
            next_obs, reward, done = game.step(action)
            episode_reward += reward

            ''' get value for next state Q(s', a') '''
            action_values = []
            for n in range(num_actions):
                action_values.append(q_table[(next_obs[0], next_obs[1], next_obs[2], next_obs[3], n)])  # FOR SINGE AGENT and TARGET

            # get max value
            _, next_act_value = max(enumerate(action_values), key=operator.itemgetter(1))  # we have the value for Q(s',a') now

            # update the q-values
            q_table[(obs[0], obs[1], obs[2], obs[3], action)] += lamda*(reward + gamma*next_act_value - current_act_value)

            obs = next_obs
            if done:
                break


        print("Episode " + str(i_episode) + ". Total reward: " + str(episode_reward))
        episode_reward_list.append(episode_reward)
