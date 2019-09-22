import numpy as np
import matplotlib.pyplot as plt


class RandomWalk:

    def __init__(self):
        self.actions = [-1, 1]
        self.initial_state = 3
        self.true_state_values = np.zeros(7)
        self.true_state_values[1:6] = np.arange(1, 6)/6.0
        self.state_values = np.zeros(7)
        self.state_values[1:6] = 0.5
        self.n_episodes = 100
        self.rmse_td = np.zeros(self.n_episodes+1)
        self.rmse_mc = np.zeros(self.n_episodes+1)

    def take_step(self, state, action):
        if action == -1:
            next_state = state - 1
        else:
            next_state = state + 1
        if next_state == 6:
            reward = 1
        else:
            reward = 0
        return next_state, reward

    def temporal_difference(self,alpha):
        current_state = self.initial_state
        while True:
            action = np.random.choice(self.actions)
            next_state, reward = self.take_step(current_state, action)
            self.state_values[current_state] += alpha * (reward + self.state_values[next_state] - self.state_values[current_state])
            current_state = next_state
            if current_state == 0 or current_state == 6:
                break

    def alpha_mc(self):
        episode = list()
        current_state = self.initial_state
        while True:
            action = np.random.choice(self.actions)
            next_state, reward = self.take_step(current_state, action)
            episode.append([current_state, reward])
            current_state = next_state
            if current_state == 0 or current_state == 6:
                break
        return episode

    def compute_values_td(self,alpha,plot_fig):
        episodes = [1, 10, 100]
        if plot_fig:
            plt.figure()
            plt.plot(self.state_values[1:6], label=0)
        for i in range(self.n_episodes+1):
            self.temporal_difference(alpha)
            self.rmse_td[i] = np.sqrt(np.sum(np.power(self.true_state_values - self.state_values, 2))/5.0)
            if i in episodes and plot_fig:
                plt.plot(self.state_values[1:6], label=i)
        if plot_fig:
            plt.plot(self.true_state_values[1:6], label='True Values')
            plt.legend()
            plt.xlabel('State')
            plt.ylabel('Estimated value')
            plt.savefig('q6_left.png')
            plt.show()
            plt.close()

    def compute_values_mc(self,alpha):
        for i in range(self.n_episodes+1):
            episode = self.alpha_mc()
            g = 0
            t = len(episode)-1
            while t >= 0:
                s_t = episode[t][0]
                r_t = episode[t][1]
                g += r_t
                self.state_values[s_t] += alpha*(g - self.state_values[s_t])
                t = t - 1
            self.rmse_mc[i] = np.sqrt(np.sum(np.power(self.true_state_values - self.state_values, 2))/5.0)

def plot_q6_right():
    n_episodes = 100
    n_runs = 100
    td_alphas = [0.1, 0.05, 0.15]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    plt.figure()
    for alpha in td_alphas:
        rmse_td = np.zeros((n_runs, n_episodes+1))
        for i in range(n_runs):
            obj = RandomWalk()
            if alpha == 0.1 and i == 0:
                plot_fig = True
            else:
                plot_fig = False
            obj.compute_values_td(alpha,plot_fig)
            rmse_td[i] = obj.rmse_td
        plt.plot(np.mean(rmse_td, axis=0), label='TD with alpha= '+str(alpha))
    for alpha in mc_alphas:
        rmse_mc = np.zeros((n_runs, n_episodes+1))
        for i in range(n_runs):
            obj = RandomWalk()
            obj.compute_values_mc(alpha)
            rmse_mc[i] = obj.rmse_mc
        plt.plot(np.mean(rmse_mc, axis=0), label='MC with alpha= '+str(alpha))
    plt.legend()
    plt.xlabel('Walks/Episodes')
    plt.ylabel('Empirical RMSE error, averaged over states')
    plt.savefig('q6_right.png')
    plt.show()
    plt.close()

plot_q6_right()













