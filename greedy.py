import random as rd
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    A Bandit is a Gaussian distribution of mean mu and standard deviation sigma
    """

    def __init__(self, id, mu, sigma):
        self.id = id
        self.mu = mu
        self.sigma = sigma

    def pull(self):
        """Returns a reward based on a gaussian distribution"""
        return np.random.normal(self.mu, self.sigma)


class Casino:
    """
    A Casino is just a collection of Bandits
    """

    def __init__(self, bandits=[]):
        self.bandits = bandits

    def size(self):
        """Returns the number of machines"""
        return len(self.bandits)

    def best_machine(self):
        """Returns the indice of the machine with the best mean"""
        assert self.size() > 0, "Casino Vide"
        i = 1
        best = 0
        mu_max = self.bandits[0].mu
        while i < self.size():
            if self.bandits[i].mu > mu_max:
                mu_max = self.bandits[i].mu
                best = i
            i += 1
        return best


class Eps_Agent:
    """
    Implementation of an agent using the epsilon-greedy method.
    Parameters :
    eps = the probability to explore at each iteration
    bandits = a list of Bandits
    initial_value = if we want to do an optimistic method.
    """

    def __init__(self, eps=0.1, bandits=[], initial_value=0):
        """
        An agent contains:
        - the values used in the policy (here eps)
        - the historic of reward and the number of time the best option was chosen in a temporal list
        - a list indicating for each bandit the actual prediction of the mean reward, and the number of time it was explored.

        """
        self.eps = eps
        self.casino = Casino(bandits)
        self.best_bandit = self.casino.best_machine()
        self.prediction = [initial_value for i in range(self.casino.size())]
        self.number_explored = [0 for i in range(self.casino.size())]
        self.rewards = []
        self.chose_the_best = []
        self.nb_eps = 0

    def choose(self):
        """
        Function that implements the action of the agent at time self.time.
        It choses a bandit, add the result at the reward list (and indicates in the chose_the_best list if the option
        chosen was the best bandit) and increment the list recensing the number of explorations of each option
        """
        exploration = np.random.binomial(1, self.eps)
        if exploration:
            bandit_id = np.random.randint(0, self.casino.size())
            self.nb_eps += 1
        else:
            bandit_id = np.argmax(self.prediction)
        self.number_explored[bandit_id] += 1
        reward = self.casino.bandits[bandit_id].pull()
        self.rewards.append(reward)
        self.prediction[bandit_id] = (
            self.prediction[bandit_id] * (self.number_explored[bandit_id] - 1) + reward
        ) / self.number_explored[bandit_id]
        self.chose_the_best.append(bandit_id == self.best_bandit)


class UCB_Agent:
    """
    Implementation of an agent using the Upper Confidence Bound method.
    Parameters :
    c = the confidence value
    bandits = a list of Bandits
    initial_value = if we want to do an optimistic method.
    """

    def __init__(self, c=0.1, bandits=[], initial_value=0):
        """
        An agent contains:
        - the values used in the policy (here confidence_value)
        - the historic of reward and the number of time the best option was chosen in a temporal list
        - a list indicating for each bandit the actual prediction of the mean reward, and the number of time it was explored.
        - a time counter.
        """
        self.confidence_value = c
        self.casino = Casino(bandits)
        self.best_bandit = self.casino.best_machine()
        self.prediction = [initial_value for i in range(self.casino.size())]
        self.number_explored = [0 for i in range(self.casino.size())]
        self.rewards = []
        self.chose_the_best = []
        self.time = 0

    def choose(self):
        """
        Function that implements the action of the agent at time self.time.
        It choses a bandit, add the result at the reward list (and indicates in the chose_the_best list if the option
        chosen was the best bandit) and increment the time counter and the list recensing the number of explorations of each option
        """
        self.time += 1
        ucb_estimation = [
            self.prediction[i]
            + self.confidence_value
            * np.sqrt(np.log(self.time) / self.number_explored[i])
            if self.number_explored[i] > 0
            else np.inf
            for i in range(self.casino.size())
        ]
        bandit_id = np.argmax(
            ucb_estimation
        )  # This method is a bit biased because in case of multiple occurences of the max value, its the indice of the first occurence that is returned.
        self.number_explored[bandit_id] += 1
        reward = self.casino.bandits[bandit_id].pull()
        self.rewards.append(reward)
        self.prediction[bandit_id] = (
            self.prediction[bandit_id] * (self.number_explored[bandit_id] - 1) + reward
        ) / self.number_explored[bandit_id]
        self.chose_the_best.append(bandit_id == self.best_bandit)


class Etude_Statistique:
    """
    Class allowing to plot the comportment of an agent (epsilon-greedy, optimistic-greedy or UCB-agent) over k iterations.
    """

    def __init__(self, k, policy_parameter, machines, ucb=False, initial_value=0):
        self.nb_iterations = k
        if not ucb:
            self.agent = Eps_Agent(policy_parameter, machines, initial_value)
        else:
            self.agent = UCB_Agent(policy_parameter, machines, initial_value)

    def run(self):
        """
        Function that make the agent choose during self.nb_iter iterations
        """
        for i in range(self.nb_iterations):
            self.agent.choose()

    def mean_reward(self):
        return [np.mean(self.agent.rewards[:i]) for i in range(self.nb_iterations)]

    def plot_mean_reward(self, title, figname):
        """
        Function plotting the mean reward depending on the number of iterations
        """
        Y = self.mean_reward
        plt.plot([i for i in range(self.nb_iterations)], Y)
        plt.xlabel("Number of iteration")
        plt.ylabel("Mean reward")
        plt.title(title)
        plt.savefig(figname)
        plt.show()

    def best_option(self):
        return [
            np.sum(self.agent.chose_the_best[:i]) / i
            for i in range(1, self.nb_iterations)
        ]

    def plot_best_option(self, title, figname):
        """
        Function to plot the frequence of the best option being chosen depending of the number of iterations.
        """
        Y = self.best_option()
        plt.plot([i for i in range(1, self.nb_iterations)], Y)
        plt.xlabel("Number of iteration")
        plt.ylabel("Percentage of best option being chosen")
        plt.title(title)
        plt.savefig(figname)
        plt.show()


class Etude_Agents:
    """
    Class regrouping nb_agents with same policy_parameters eps and initial_value, and working on same Bandits machines
    Allows to average out the comportment of a type of agent (epsilon-greedy, optimistic-greedy or UCb-agent)
    """

    def __init__(
        self, nb_agents, k, policy_parameter, machines, ucb=False, initial_value=0
    ):
        self.nb_agents = nb_agents
        self.agents = []
        for i in range(nb_agents):
            if not ucb:
                self.agents.append(Eps_Agent(policy_parameter, machines, initial_value))
            else:
                self.agents.append(UCB_Agent(policy_parameter, machines, initial_value))
        self.nb_iter = k

    def run(self):
        """
        Function that make each agent choose during self.nb_iter iterations
        """
        for i in range(self.nb_iter):
            for k in range(self.nb_agents):
                self.agents[k].choose()

    def mean_reward(self):
        return [
            np.mean(
                [np.mean(self.agents[k].rewards[:i]) for k in range(self.nb_agents)]
            )
            for i in range(1, self.nb_iter)
        ]

    def plot_mean_reward(self, title, figname):
        """
        Function plotting the mean reward depending on the number of iterations
        """
        Y = self.mean_reward()
        plt.plot([i for i in range(1, self.nb_iter)], Y)
        plt.xlabel("Number of iteration")
        plt.ylabel("Mean reward")
        plt.title(title)
        plt.savefig(figname)
        plt.show()

    def best_option(self):
        return [
            np.mean(
                [
                    np.sum(self.agents[k].chose_the_best[:i]) / i
                    for k in range(self.nb_agents)
                ]
            )
            for i in range(1, self.nb_iter)
        ]

    def plot_best_option(self, title, figname):
        """
        Function to plot the frequence of the best option being chosen depending of the number of iterations.
        """
        Y = self.best_option()
        plt.plot([i for i in range(1, self.nb_iter)], Y)
        plt.xlabel("Number of iteration")
        plt.ylabel("Percentage of best option being chosen")
        plt.title(title)
        plt.savefig(figname)
        plt.show()


class comparaisonMethodes:
    def __init__(self, nb_agents, k, machines, eps=0.1, c=0.1, initial_value=8):
        self.epsilon_greedy = Etude_Agents(nb_agents, k, eps, machines)
        self.optimistic_greedy = Etude_Agents(
            nb_agents, k, eps, machines, initial_value=initial_value
        )
        self.ucb_agent = Etude_Agents(
            nb_agents, k, c, machines, ucb=True, initial_value=initial_value
        )
        self.nb_iter = k
        self.q0 = initial_value

    def run(self):
        """
        Function that make each agent choose during self.nb_iter iterations
        """
        self.optimistic_greedy.run()
        self.epsilon_greedy.run()
        self.ucb_agent.run()

    def plot_comparaisons_mean_reward(self, title, figname):
        x = [i for i in range(1, self.nb_iter)]
        y1 = self.epsilon_greedy.mean_reward()
        y2 = self.optimistic_greedy.mean_reward()
        y3 = self.ucb_agent.mean_reward()
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.plot(x, y3)
        plt.xlabel("Number of iterations")
        plt.ylabel("Mean reward")
        plt.title(title)
        plt.legend(
            "eps_greedy = " + str(self.epsilon_greedy.agents[0].eps),
            "opt_greedy with same eps and q0 =" + str(self.q0),
            "ucb with c= " + str(self.ucb_agent.agents[0].confidence_value),
        )
        plt.savefig(figname)

    def plot_comparaisons_best_option(self, title, figname):
        x = [i for i in range(1, self.nb_iter)]
        y1 = self.epsilon_greedy.best_option()
        y2 = self.optimistic_greedy.best_option()
        y3 = self.ucb_agent.best_option()
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.plot(x, y3)
        plt.xlabel("Number of iterations")
        plt.ylabel("Accuracy (in %)")
        plt.title(title)
        plt.legend(
            "eps_greedy = " + str(self.epsilon_greedy.agents[0].eps),
            "opt_greedy with same eps and q0 =" + str(self.q0),
            "ucb with c= " + str(self.ucb_agent.agents[0].c),
        )
        plt.savefig(figname)
