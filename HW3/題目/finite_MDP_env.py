import numpy as np
import random
import gym.spaces as sp

class environment():
    def __init__(self):
        self.action_space = sp.Discrete(2)
        self.observation_space = sp.Discrete(8)
        self.s = 0
        self.done = False
        self.transition = {0: [1, 2], 1: [3, 4], 2: [4, 5], 4: [6], 5: [6, 7]}
        self.trans_probability = {0: {0: 0.2, 1: 0.7, 2: 0.3, 4: 1, 5: 0.2},
                                  1: {0: 0.9, 1: 0.2, 2: 0.7, 4: 1, 5: 0.8}}
        self.trans_reward = {0: [15, 7], 1: [7, -15], 2: [-20, 5], 4: [30], 5: [0, 10]}
        self.done_list = [3, 6, 7]
        self.render_state_list = ['S0', 'S1', 'S2', 'T1', 'S3', 'S4', 'T2', 'T3']

    def step(self, a):
        if not self.done:
            s = self.s
            p = self.trans_probability[a][s]
            s_index = 0 if random.random() <= p else 1
            s_ = self.transition[s][s_index]
            r = self.trans_reward[s][s_index]
            d = s_ in self.done_list
            self.s = s_
            self.done = d
            return self.state, r, d, None
        else:
            raise StopIteration('Current environment is done.')

    @property
    def state(self):
        s = np.zeros(8)
        s[self.s] = 1
        return s

    @state.getter
    def state(self):
        s = np.zeros(8)
        s[self.s] = 1

        return s


    def reset(self):
        self.s = 0
        self.done = False
        return self.state

    def render(self):
        print("current state is", self.render_state_list[self.s])






