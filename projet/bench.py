import numpy as np
import random
import operator

class Agent:
    def __init__(self):

        self.reward = random.random()*100
        # if w:
        #     self.NN = NN.from_weights(w)
        # else:
        #     self.NN = NN.from_params(self.observation_size, [2], self.action_size)

    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action

    def set_reward(self, reward):
        self.reward = reward

agents = [Agent() for i in range(50)]
agents.sort(key=operator.attrgetter('reward'))
breakpoint()