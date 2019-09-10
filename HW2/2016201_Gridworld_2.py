#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class GridWorld:

    # Initialization of parameters, variable names are self-explanatory
    def __init__(self, no_of_rows, no_of_cols, discount):

        self.no_of_rows = no_of_rows
        self.no_of_cols = no_of_cols
        self.discount_rate = discount
        self.v_pi = np.zeros((no_of_rows, no_of_cols))
        self.no_of_states = no_of_rows*no_of_cols
        self.actions = list()
        self.action_map = dict()
        self.actions.append([0, -1])  # West action
        self.actions.append([-1, 0])  # North action
        self.actions.append([0, 1])  # East action
        self.actions.append([1, 0])  # South action
        self.action_map[0] = "W"
        self.action_map[1] = "N"
        self.action_map[2] = "E"
        self.action_map[3] = "S"
        self.prob_matrix = 0.25 * np.ones((self.no_of_rows, self.no_of_cols, len(self.actions)))

    # Returns the 1D coordinate corresponding to the state number given its coordinates in state matrix
    def get_1d_state_coord(self, state):

        return state[0] * self.no_of_rows + state[1]

    def take_step(self,state,action):

        next_state = list()
        next_state.append(state[0]+action[0])
        next_state.append(state[1]+action[1])
        if next_state[0] < 0 or next_state[0] >= self.no_of_rows or next_state[1] < 0 or next_state[1] >= self.no_of_cols:  # Off the grid locations
            next_state = state
        return next_state, -1

    def policy_iteration(self):
        flag = 0
        iter_no = 0
        while flag == 0:
            delta = 0
            old_v_s = np.copy(self.v_pi)
            for i in range(self.no_of_rows):
                for j in range(self.no_of_cols):
                    if i == 0 and j == 0:
                        continue
                    elif i == self.no_of_rows - 1 and j == self.no_of_cols - 1:
                        continue
                    old_value = old_v_s[i, j]
                    new_value = 0
                    for k in range(len(self.actions)):
                        next_state, reward = self.take_step([i, j], self.actions[k])
                        new_value += self.prob_matrix[i, j, k] * (reward + self.discount_rate * self.v_pi[next_state[0], next_state[1]])
                    self.v_pi[i, j] = new_value
                    delta = max(delta, abs(old_value - new_value))

            diff = np.sum(np.abs(old_v_s - self.v_pi))
            iter_no += 1
            print("Iteration number: "+str(iter_no))
            print(np.round(self.v_pi,1))
            print("Change in value function"+": "+str(diff))
            if delta < 0.0001:
                flag = 1

            optimal_policy = dict()
            policy_stable = True
            for i in range(self.no_of_rows):
                for j in range(self.no_of_cols):
                    old_pi = np.copy(self.prob_matrix[i, j])
                    action_value = np.zeros(len(self.actions))
                    for k in range(len(self.actions)):
                        next_state, reward = self.take_step([i, j], self.actions[k])
                        action_value[k] = self.v_pi[next_state[0], next_state[1]]
                    max_value = np.max(action_value)
                    optimal_actions = list()
                    for r in range(len(self.actions)):
                        if action_value[r] == max_value:
                            optimal_actions.append(self.action_map[r])
                    optimal_policy[self.get_1d_state_coord([i, j])] = optimal_actions
                    for s in range(len(self.actions)):
                        if action_value[s] == max_value:
                            self.prob_matrix[i, j, s] = 1/len(optimal_actions)
                        else:
                            self.prob_matrix[i, j, s] = 0
                    for w in range(len(self.actions)):
                        if old_pi[w] != self.prob_matrix[i, j, w]:
                            policy_stable = False
                            break
            temp_count = 0
            print("Optimal Policy")
            for i in range(self.no_of_rows):
                for j in range(self.no_of_cols):
                    if i == 0 and j == 0:
                        print("--"+" | ", end=" ")
                    elif i == self.no_of_rows - 1 and j == self.no_of_cols - 1:
                        print("--"+" | ", end=" ")
                    else:
                        print(str(optimal_policy[temp_count]) + " | ", end=" ")
                    temp_count += 1
                print()
            print()
            if policy_stable == True:
                flag = 1
        return self.v_pi, optimal_policy

    def find_optimal_policy(self, value_func):

        optimal_policy = dict()
        for i in range(self.no_of_rows):
            for j in range(self.no_of_cols):
                action_value = np.zeros(len(self.actions))
                for k in range(len(self.actions)):
                    next_state, reward = self.take_step([i, j], self.actions[k])
                    action_value[k] = value_func[next_state[0], next_state[1]]
                max_value = np.max(action_value)
                optimal_actions = list()
                for k in range(len(self.actions)):
                    if action_value[k] == max_value:
                        optimal_actions.append(self.action_map[k])
                optimal_policy[self.get_1d_state_coord([i, j])] = optimal_actions
        return optimal_policy

    def value_iteration(self):
        flag = 0
        iter_no = 0
        while flag == 0:
            delta = 0
            old_v_s = np.copy(self.v_pi)
            for i in range(self.no_of_rows):
                for j in range(self.no_of_cols):
                    if i == 0 and j == 0:
                        continue
                    elif i == self.no_of_rows - 1 and j == self.no_of_cols - 1:
                        continue
                    old_value = old_v_s[i, j]
                    new_value = - float('inf')
                    for k in range(len(self.actions)):
                        next_state, reward = self.take_step([i, j], self.actions[k])
                        new_value = max(new_value, reward + self.discount_rate * self.v_pi[next_state[0], next_state[1]])
                    delta = max(delta, abs(old_value - new_value))
                    self.v_pi[i, j] = new_value
            diff = np.sum(np.abs(old_v_s - self.v_pi))
            iter_no += 1
            print("Iteration number: " + str(iter_no))
            print(np.round(self.v_pi, 1))
            print("Change in value function" + ": " + str(diff))
            print()
            if delta < 0.0001:
                flag = 1
        optimal_policy = self.find_optimal_policy(self.v_pi)
        return self.v_pi, optimal_policy


# In[3]:


# Question 6

# Policy iteration

print("Policy Iteration")
print()
gridworld_obj = GridWorld(4, 4, 1)
vpi, policy = gridworld_obj.policy_iteration()
print("Final Value function after Policy Iteration")
print(vpi)
print("Final Optimal Policy after Policy Iteration")
temp_c = 0
for a in range(gridworld_obj.no_of_rows):
    for b in range(gridworld_obj.no_of_cols):
        if a == 0 and b == 0:
            print("--" + " | ", end=" ")
        elif a == 3 and b == 3:
            print("--" + " | ", end=" ")
        else:
            print(str(policy[temp_c]) + " | ", end=" ")
        temp_c += 1
    print()
print()


# In[4]:


# Value iteration

print("Value Iteration")
print()
gridworld_obj = GridWorld(4, 4, 1)
vpi, policy = gridworld_obj.value_iteration()
print("Final Value function after Value Iteration")
print(vpi)
print("Final Optimal Policy after Policy Iteration")
temp_c = 0
for a in range(gridworld_obj.no_of_rows):
    for b in range(gridworld_obj.no_of_cols):
        if a == 0 and b == 0:
            print("--" + " | ", end=" ")
        elif a == 3 and b == 3:
            print("--" + " | ", end=" ")
        else:
            print(str(policy[temp_c]) + " | ", end=" ")
        temp_c += 1
    print()
print()

