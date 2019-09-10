#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns


# In[18]:


class Jack_Car_Rental:

    def __init__(self,parking_cost,no_of_free_rides):
        self.rent_reward = 10
        self.move_cost = 2
        self.no_of_locations = 2
        self.expected_rent_requests = [3, 4]
        self.expected_returns = [3, 2]
        self.discount_rate = 0.9
        self.maximum_cars = [21, 21]
        self.policy = np.zeros(self.maximum_cars, dtype= np.int)
        self.values = np.zeros(self.maximum_cars)
        self.actions = list()
        self.max_moves = 5
        for i in range(-5,6):
            self.actions.append(i)
        self.prob_map = {}
        self.reward_map = {}
        self.parking_cost = parking_cost
        self.no_of_free_rides = no_of_free_rides
        for w in range(2):
            self.prob_map[w] = np.zeros([self.maximum_cars[w] + self.max_moves, self.maximum_cars[w]])
            self.reward_map[w] = np.zeros(self.maximum_cars[w] + self.max_moves)
        self.setup()

    def setup(self):

        for i in range(self.no_of_locations):
            no_of_reqs = 0
            while True:
                prob_reqs = self.poisson_pmf(no_of_reqs, self.expected_rent_requests[i])
                if prob_reqs < 0.000001:
                    break
                a = 0
                while a < self.reward_map[i].shape[0]:
                    self.reward_map[i][a] += self.rent_reward * prob_reqs * min(no_of_reqs, a)
                    a = a + 1
                drop_offs = 0
                while True:
                    prob_drop = self.poisson_pmf(drop_offs, self.expected_returns[i])
                    if prob_drop < 0.000001:
                        break
                    a = 0
                    while a < self.prob_map[i].shape[0]:
                        fulfilled_req = min(no_of_reqs,a)
                        temp = min(self.maximum_cars[i]-1,(a + drop_offs - fulfilled_req))
                        a_new = max(0,temp)
                        self.prob_map[i][a, a_new] += prob_reqs * prob_drop
                        a = a + 1
                    drop_offs = drop_offs + 1
                no_of_reqs = no_of_reqs + 1

    def policy_evaluation(self):

        while True:
            delta = 0
            i = 0
            while i < self.maximum_cars[0]:
                j = 0
                while j < self.maximum_cars[1]:
                    old_v_pi = np.copy(self.values[i, j])
                    action = self.policy[i, j]
                    self.values[i, j] = self.take_step(i, j, action)
                    delta = max(delta, abs(old_v_pi - self.values[i, j]))
                    j = j + 1
                i = i + 1
            if delta < 0.000000001:
                break

    def find_optimal_action(self,i,j):

        max_value = - float('inf')
        optimal_action = self.policy[i,j]
        lower_bound = max(self.actions[0], -j)
        upper_bound = min(self.actions[-1], i)+1
        for action in range(lower_bound,upper_bound):
            new_value = self.take_step(i,j,action)
            if new_value - max_value > 0.000000001:
                max_value = new_value
                optimal_action = action
        return optimal_action

    def is_stable_policy(self):

        policy_stable = True
        i = 0
        while i < self.maximum_cars[0]:
            j = 0
            while j < self.maximum_cars[1]:
                optimal_action = self.find_optimal_action(i, j)
                if self.policy[i, j] != optimal_action:
                    self.policy[i,j] = optimal_action
                    policy_stable = False
                j = j + 1
            i = i + 1
        return policy_stable

    def policy_iteration(self):

        iter_no = 0
        past_policies = []
        policy_stable = False
        self.plot_policy_function(iter_no)
        while policy_stable == False:
            old_pi = np.copy(self.policy)
            past_policies.append(old_pi)
            self.policy_evaluation()
            iter_no += 1
            if self.is_stable_policy() == False:
                self.plot_policy_function(iter_no)
            for old_pol in past_policies:
                if self.is_policy_equal(self.policy,old_pol):
                    policy_stable = True
                    self.plot_value_function()
                    break

    def is_policy_equal(self,p1,p2):

        for i in range(self.maximum_cars[0]):
            for j in range(self.maximum_cars[1]):
                if p1[i,j] != p2[i,j]:
                    return False
        return True

    def take_step(self, x1, x2, a):

        t1 = min(a, x1)
        a = max(t1, -1 * x2)
        t2 = min(self.actions[-1], a)
        a = max(self.actions[0], t2)
        val = -1 * abs(a) * self.move_cost
        if x1 - a > 10:
            val = val - self.parking_cost
        if x2 + a > 10:
            val = val - self.parking_cost
        if a > 0:
            if a <= self.no_of_free_rides:
                val = val + self.move_cost * a
            else:
                val = val + self.move_cost * self.no_of_free_rides
        p = 0
        while p < self.maximum_cars[0]:
            q = 0
            while q < self.maximum_cars[1]:
                total_reward = self.reward_map[0][x1 - a] + self.reward_map[1][x2 + a]
                val = val + (self.prob_map[0][x1 - a, p] * self.prob_map[1][x2 + a, q] * (total_reward + self.discount_rate * self.values[p, q]))
                q = q + 1
            p = p + 1
        return val

    def poisson_pmf(self,n,lambda_val):
        return stats.poisson.pmf(n, lambda_val)

    def plot_policy_function(self,iter):
        y_axis = np.arange(self.maximum_cars[0] - 1, -1, -1)
        plt.figure()
        sns.heatmap(np.flipud(self.policy), cmap="bwr", vmin=min(self.actions), vmax=max(self.actions),
                    yticklabels=y_axis, annot=True, cbar=None)
        plt.title("Policy function at iteration: " + str(iter))
        plt.xlabel("#Cars at second location")
        plt.ylabel("#Cars at first location")
        plt.show()
        plt.close()

    def plot_value_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(self.values.shape[0])
        y = np.arange(self.values.shape[1])
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, self.values)
        ax.set_xlabel('Cars at location 1')
        ax.set_ylabel('Cars at location 2')
        ax.set_zlabel('State value')
        plt.show()
        
        
        


# In[19]:


obj = Jack_Car_Rental(0,0)
obj.policy_iteration()


# In[20]:


obj = Jack_Car_Rental(4,1)
obj.policy_iteration()

