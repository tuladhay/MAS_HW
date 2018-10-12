import random


class Agent:
    def __init__(self, width=10, height=5):
        self.pos_x = 0
        self.pos_y = 0
        self.policy = None  # DDPG class
        self.action = None
        self.width = width - 1  # grid positions are from 0 - 9
        self.height = height - 1

    def compute_action(self):
        # TODO: REPLACE THIS WITH A LEARNING ALGORITHM
        action = random.randint(0,3)
        self.action = action

    def update_pos(self):
        if self.action == 0:  # left
            if self.pos_x != 0:  # already at the leftmost
                self.pos_x -= 1
            else:
                pass
        elif self.action == 1:  # right
            if self.pos_x != self.width:
                self.pos_x += 1
            else:
                pass
        elif self.action == 2:  # up
            if self.pos_y != self.height:
                self.pos_y += 1
            else:
                pass
        elif self.action == 3:  # down
            if self.pos_y != 0:
                self.pos_y -= 1
            else:
                pass
        else:
            print("invalid action")

    def reset(self):
        self.pos_x = 0
        self.pos_y = 0


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
        self.pos_x = 0
        self.pos_y = 0
