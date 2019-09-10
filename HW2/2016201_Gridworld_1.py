#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import optimize


# In[2]:


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
        self.action_map = dict()
        self.actions.append([0, -1])  # West action
        self.actions.append([-1, 0])  # North action
        self.actions.append([0, 1])  # East action
        self.actions.append([1, 0])  # South action
        self.action_map[0] = "W"
        self.action_map[1] = "N"
        self.action_map[2] = "E"
        self.action_map[3] = "S"

    # Returns the 2D coordinate corresponding to the state matrix given the state number
    def get_2d_state_coord(self,state):

        x_coord = state // self.no_of_rows
        y_coord = state % self.no_of_rows
        return [x_coord, y_coord]

    # Returns the 1D coordinate corresponding to the state number given its coordinates in state matrix
    def get_1d_state_coord(self,state):

        return state[0]*self.no_of_rows + state[1]

    # Finds state-value function by solving linear equation Ax = B
    def solve_linear_equations(self):
        # A_matrix : Matrix of 25*25 dimension consisting of coefficients of the value function for each state
        # B_matrix : Vector of 25*1 dimension consisting of the constant term for each linear equation

        A_matrix = np.identity(self.no_of_states)
        B_matrix = np.zeros(self.no_of_states)

        # Finding the coefficient of state-value function for each state
        for i in range(self.no_of_states):
            for action in self.actions:
                next_state, reward = self.take_step(self.get_2d_state_coord(i), action)
                next_state = self.get_1d_state_coord(next_state)
                A_matrix[i, next_state] = A_matrix[i, next_state] - self.discount_rate*self.action_prob
                B_matrix[i] = B_matrix[i] + self.action_prob*reward

        # Solving the system of equations using linear algebra
        inverse_of_A = np.linalg.inv(A_matrix)
        value_function = np.dot(inverse_of_A, B_matrix)
        value_function = np.round(value_function, 1)
        value_function = value_function.reshape((self.no_of_rows, self.no_of_cols))
        print(value_function)

    # Finds non-optimal state-value function using iterations
    def find_v_pi(self):

        flag = 0
        while flag == 0:  # Until convergence
            v_pi_new = np.zeros(self.v_pi.shape)
            for i in range(self.no_of_rows):  # Iterate over states
                for j in range(self.no_of_cols):
                    for action in self.actions:  # Iterate over actions
                        next_state, reward = self.take_step([i, j], action)  # Take step and get reward
                        v_pi_new[i, j] += self.action_prob * (reward + self.discount_rate * self.v_pi[next_state[0], next_state[1]])  # Bellman's update rule
            if np.sum(np.abs(self.v_pi-v_pi_new)) < 0.0004:  # Check convergence
                print(np.round(self.v_pi,1))
                break
            self.v_pi = v_pi_new  # Update state-value function

    # Takes current state and action taken as parameters and returns next state and reward
    def take_step(self, state, action):

        if state[0] == 0 and state[1] == 1:  # Check whether state is A
            return [4, 1], 10
        elif state[0] == 0 and state[1] == 3:  # Check whether state is B
            return [2, 3], 5
        next_state = list()
        next_state.append(state[0]+action[0])
        next_state.append(state[1]+action[1])
        if next_state[0] < 0 or next_state[0] >= self.no_of_rows or next_state[1] < 0 or next_state[1] >= self.no_of_cols:  # Off the grid locations
            reward = -1
            next_state = state
        else:
            reward = 0
        return next_state, reward

    def optimize_non_linear_system(self, A, B):
        scipy_obj = optimize.linprog(c = np.ones(self.no_of_states), A_ub = A, b_ub = B)
        value_function = scipy_obj.x
        return value_function

    def find_optimal_v_pi(self):

        A_matrix_row_dim = len(self.actions) * self.no_of_states
        A_matrix_col_dim = self.no_of_states
        A_matrix = np.zeros((A_matrix_row_dim, A_matrix_col_dim))
        B_matrix = np.zeros(A_matrix_row_dim)

        for i in range(self.no_of_states):
            for j in range(len(self.actions)):
                next_state, reward = self.take_step(self.get_2d_state_coord(i), self.actions[j])
                next_state = self.get_1d_state_coord(next_state)
                index = len(self.actions) * i + j
                A_matrix[index, i] = -1
                A_matrix[index, next_state] = A_matrix[index, next_state] + self.discount_rate
                B_matrix[index] = B_matrix[index] - reward
        optimal_v_pi = self.optimize_non_linear_system(A_matrix, B_matrix)
        return np.round(optimal_v_pi, 1)

    def find_optimal_policy(self, value_func):

        optimal_policy = dict()
        for i in range(self.no_of_states):
            action_value = np.zeros(len(self.actions))
            for j in range(len(self.actions)):
                next_state, reward = self.take_step(self.get_2d_state_coord(i), self.actions[j])
                next_state = self.get_1d_state_coord(next_state)
                action_value[j] = value_func[next_state]
            max_value = np.max(action_value)
            optimal_actions = list()
            for k in range(len(self.actions)):
                if action_value[k] == max_value:
                    optimal_actions.append(self.action_map[k])
            optimal_policy[i] = optimal_actions
        return optimal_policy


# In[3]:


# Question 2

gridworld_obj = GridWorld(5, 5, 0.9, 0.25)

print("Question 1 - Value function")
print("Using system of equations")
gridworld_obj.solve_linear_equations() # Using system of equations
print("Using iterations")
gridworld_obj.find_v_pi() # Using iterations


# In[4]:


# Question 4

gridworld_obj = GridWorld(5, 5, 0.9, 0.25)

optimal_value_func = gridworld_obj.find_optimal_v_pi()
optimal_policy = gridworld_obj.find_optimal_policy(optimal_value_func)

print("Optimal value function")
print(np.reshape(optimal_value_func,(gridworld_obj.no_of_rows, gridworld_obj.no_of_cols)))
print("Optimal policy")
temp_count = 0
for i in range(gridworld_obj.no_of_rows):
    for j in range(gridworld_obj.no_of_cols):
        print(str(optimal_policy[temp_count])+" | ",end=" ")
        temp_count += 1
    print()

