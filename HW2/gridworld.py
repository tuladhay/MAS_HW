import numpy as np
import random
from operator import attrgetter
from entities import Agent, Target
import torch
import operator


class Gridworld():
    def __init__(self, width, height, n_agents):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.agents = [Agent() for _ in range(n_agents)]  # initialize multiple agents
        self.target = Target()
        self.grid = np.zeros([self.width, self.height])
        self.timestep = 0
        self.done = False

        self.set_agent_pos()
        self.set_target_pos()

        self.obs = []

    def set_agent_pos(self):
        for a in self.agents:
            pos_x = a.pos_x
            pos_y = a.pos_y
            self.grid[pos_x][pos_y] = 1

    def set_target_pos(self, random_pos=False):
        if random_pos:
            pos_x = random.randint(0, self.width)
            pos_y = random.randint(0, self.height)
        else:
            pos_x = self.target.pos_x
            pos_y = self.target.pos_y
        self.grid[pos_x][pos_y] = 8

    def step(self, obs):
        agent_rewards = []
        reward = 0
        reward -= 0.5  # negative reward for each step
        # generate next action
        for a in self.agents:
            a.compute_action(obs)
            a.update_pos()

        self.target.move_random()
        # update positions in the grid world
        self.grid_reset()
        self.set_agent_pos()
        self.set_target_pos(random_pos=False)

        for a in self.agents:
            if a.pos_x == self.target.pos_x and a.pos_y == self.target.pos_y:
                agent_rewards.append(20)
                self.done = True
            else:
                agent_rewards.append(0)

        self.timestep += 1
        print("reward : "+str(reward)+str("\n"))

        # create observation vector of agent position(s) and target position
        observations = []
        for a in self.agents:
            observations.append(a.pos_x)
            observations.append(a.pos_y)
        observations.append(self.target.pos_x)
        observations.append(self.target.pos_y)

        return observations, agent_rewards, self.done

    def reset(self):
        self.timestep = 0
        for a in self.agents:
            a.reset()
        self.target.reset()

        observations = []
        for a in self.agents:
            observations.append(a.pos_x)
            observations.append(a.pos_y)
        observations.append(self.target.pos_x)
        observations.append(self.target.pos_y)
        return observations

    def grid_reset(self):
        self.grid = np.zeros([self.width, self.height])


if __name__ == "__main__":
    game = Gridworld(10, 5, 1)
    obs = torch.Tensor(game.reset())
    action = game.agents[0].compute_action(obs)  #softmax
    action_converted = action.numpy()
    index, value = max(enumerate(action_converted), key=operator.itemgetter(1))
    action_converted = index
    obs, reward, done = game.step(action_converted)
