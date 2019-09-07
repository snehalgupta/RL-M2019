import numpy as np


class GridWorld:

    # Initialization of parameters, variable names are self-explanatory
    def __init__(self, no_of_rows, no_of_cols, discount, prob):

        self.no_of_rows = no_of_rows
        self.no_of_cols = no_of_cols
        self.discount_rate = discount
        self.action_prob = prob
        self.v_pi = np.zeros((no_of_rows, no_of_cols))
        self.no_of_states = no_of_rows*no_of_cols
        self.actions = list()
        self.actions.append([0, -1])
        self.actions.append([-1, 0])
        self.actions.append([0, 1])
        self.actions.append([1, 0])

    # Finds state-value function
    def find_v_pi(self):

        flag = 0
        while flag == 0: # Until convergence
            v_pi_new = np.zeros(self.v_pi.shape)
            for i in range(self.no_of_rows): # Iterate over states
                for j in range(self.no_of_cols): # Iterate over actions
                    for action in self.actions:
                        next_state, reward = self.take_step([i, j], action) # Take step and get reward
                        v_pi_new[i, j] += self.action_prob * (reward + self.discount_rate * self.v_pi[next_state[0], next_state[1]]) # Bellman's update rule
            if np.sum(np.abs(self.v_pi-v_pi_new)) < 0.0004: # Check convergence
                print(self.v_pi)
                break
            self.v_pi = v_pi_new

    # Takes current state and action taken as parameters and returns next state and reward
    def take_step(self, state, action):

        if state[0] == 0 and state[1] == 1:
            return [4,1], 10
        elif state[0] == 0 and state[1] == 3:
            return [2,3], 5
        next_state = list()
        next_state.append(state[0]+action[0])
        next_state.append(state[1]+action[1])
        if next_state[0] < 0 or next_state[0] >= self.no_of_rows or next_state[1] < 0 or next_state[1] >= self.no_of_cols:
            reward = -1
            next_state = state
        else:
            reward = 0
        return next_state, reward

gridworld_obj = GridWorld(5,5,0.9,0.25)
gridworld_obj.find_v_pi()




