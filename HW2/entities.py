import random
import algorithms


class Agent:
    def __init__(self, width=10, height=5):
        self.pos_x = 4
        self.pos_y = 4
        # for actor critic
        self.policy = algorithms.DDPG(0.99, 0.01, 32, 4, 4)
        self.action = None
        self.width = width - 1  # grid positions are from 0 - 9
        self.height = height - 1
        self.verbose = False

    def compute_action(self, obs):
        action = self.policy.select_action(obs)
        return action

    def update_params(self, batch):
        self.policy.update_parameters(batch)

    def update_pos(self, action):
        if action == 0:  # up
            if self.pos_x != 0:  # already at the leftmost
                self.pos_x -= 1
                if self.verbose:
                    print("up")
            else:
                pass
        elif action == 1:  # down
            if self.pos_x != self.width:
                self.pos_x += 1
                if self.verbose:
                    print("down")
            else:
                pass
        elif action == 2:  # right
            if self.pos_y != self.height:
                self.pos_y += 1
                if self.verbose:
                    print("right")
            else:
                pass
        elif action == 3:  # left
            if self.pos_y != 0:
                self.pos_y -= 1
                if self.verbose:
                    print("left")
            else:
                pass
        else:
            print("invalid action")

    def reset(self, x=4, y=4, use_given_start=False):  # Option to use starting position of other agent
        if use_given_start:
            self.pos_x = x
            self.pos_y = y
        else:
            self.pos_x = random.randint(0, self.width)
            self.pos_y = random.randint(0, self.height)


class Target:
    def __init__(self, width_max=10, height_max=5):
        self.pos_x = 9
        self.pos_y = 0
        self.width = width_max - 1
        self.height = height_max - 1

    def move_random(self):
        direction = random.choice([0,1,2,3])  #left,right,up,down
        if direction == 0:  # left
            if self.pos_x != 0:  # already at the leftmost
                self.pos_x -= 1
            else:
                pass  #just stays there
        elif direction == 1:  # right
            if self.pos_x != self.width:
                self.pos_x += 1
            else:
                pass
        elif direction == 2:  # up
            if self.pos_y != self.height:
                self.pos_y += 1
            else:
                pass
        elif direction == 3:  # down
            if self.pos_y != 0:
                self.pos_y -= 1
            else:
                pass
        else:
            print("invalid action")

    def reset(self):
        self.pos_x = 9
        self.pos_y = 0
