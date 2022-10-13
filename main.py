from mimetypes import init
from greedy import *
from machines import *


def etude_unique_epsilon(eps=0.1, horizon=500):
    etude = Etude_Statistique(horizon, eps, BANDITS)
    etude.run()
    etude.mean_reward(
        title="Mean reward for a single epsilon greedy agent with eps="
        + str(eps)
        + " and with "
        + str(len(BANDITS))
        + " bandits",
        figname="mean_reward_one_agent_eps_" + str(eps)[2:] + ".jpg",
    )
    etude.best_option(
        title="Accuracy of a single epsilon greedy agent with eps="
        + str(eps)
        + " and with "
        + str(len(BANDITS))
        + " bandits",
        figname="best_option_one_agent_eps_" + str(eps)[2:] + ".jpg",
    )
    print(etude.agent.number_explored[etude.agent.best_bandit] / etude.nb_iterations)
    print(etude.agent.nb_eps / etude.nb_iterations)


def etude_moyenne_agents_epsilon(eps=0.1, horizon=500, nb_agents=1000):
    sim = Etude_Agents(nb_agents, horizon, eps, BANDITS)
    sim.run()
    sim.mean_reward(
        title="Mean reward for a group of epsilon greedy agents with eps="
        + str(eps)
        + " with "
        + str(len(BANDITS))
        + " bandits",
        figname="mean_reward_mean_agent_eps_" + str(eps)[2:] + ".jpg",
    )
    sim.best_option(
        title="Accuracy of a group of epsilon greedy agents with eps="
        + str(eps)
        + " with "
        + str(len(BANDITS))
        + " bandits",
        figname="best_option_mean_agent_eps_" + str(eps)[2:] + ".jpg",
    )


def etude_unique_optimistic(eps=0.1, horizon=500, initial_value=8):
    etude = Etude_Statistique(horizon, eps, BANDITS, initial_value=initial_value)
    etude.run()
    etude.mean_reward(
        title="Mean reward for a single optimistic greedy agent with eps="
        + str(eps)
        + "Q0 ="
        + str(initial_value)
        + " and with "
        + str(len(BANDITS))
        + " bandits",
        figname="mean_reward_one_agent_opt_" + str(eps)[2:] + ".jpg",
    )
    etude.best_option(
        title="Accuracy of a single optimistic greedy agent with eps="
        + str(eps)
        + "Q0 ="
        + str(initial_value)
        + " and with "
        + str(len(BANDITS))
        + " bandits",
        figname="best_option_one_agent_opt_" + str(eps)[2:] + ".jpg",
    )
    print(etude.agent.number_explored[etude.agent.best_bandit] / etude.nb_iterations)
    print(etude.agent.nb_eps / etude.nb_iterations)


def etude_moyenne_agents_optimistic(
    eps=0.1, horizon=500, nb_agents=1000, initial_value=8
):
    sim = Etude_Agents(nb_agents, horizon, eps, BANDITS, initial_value=initial_value)
    sim.run()
    sim.mean_reward(
        title="Mean reward for a group of optimistic greedy agents with eps="
        + str(eps)
        + "Q0 ="
        + str(initial_value)
        + " with "
        + str(len(BANDITS))
        + " bandits",
        figname="mean_reward_mean_agent_opt_" + str(eps)[2:] + ".jpg",
    )
    sim.best_option(
        title="Accuracy of a group of optimistic greedy agents with eps="
        + str(eps)
        + "Q0 ="
        + str(initial_value)
        + " with "
        + str(len(BANDITS))
        + " bandits",
        figname="best_option_mean_agent_opt_" + str(eps)[2:] + ".jpg",
    )


def etude_unique_ucb(c=0.1, horizon=500):
    etude = Etude_Statistique(horizon, c, BANDITS, ucb=True)
    etude.run()
    etude.mean_reward(
        title="Mean reward for a single ucb agent with c="
        + str(c)
        + " and with "
        + str(len(BANDITS))
        + " bandits",
        figname="mean_reward_one_agent_ucb_" + str(c)[2:] + ".jpg",
    )
    etude.best_option(
        title="Accuracy of a single ucb agent with c="
        + str(c)
        + " and with "
        + str(len(BANDITS))
        + " bandits",
        figname="best_option_one_agent_ucb_" + str(c)[2:] + ".jpg",
    )
    print(etude.agent.number_explored[etude.agent.best_bandit] / etude.nb_iterations)
    print(etude.agent.nb_eps / etude.nb_iterations)


def etude_moyenne_agents_ucb(c=0.1, horizon=500, nb_agents=1000):
    sim = Etude_Agents(nb_agents, horizon, c, BANDITS, ucb=True)
    sim.run()
    sim.mean_reward(
        title="Mean reward for a group of ucb agents with c="
        + str(c)
        + " with "
        + str(len(BANDITS))
        + " bandits",
        figname="mean_reward_mean_agent_ucb_" + str(c)[2:] + ".jpg",
    )
    sim.best_option(
        title="Accuracy of a group of ucb agents with eps="
        + str(c)
        + " with "
        + str(len(BANDITS))
        + " bandits",
        figname="best_option_mean_agent_ucb_" + str(c)[2:] + ".jpg",
    )


def comparaison(eps=0.1, c=0.1, initial_value=8, horizon=500, nb_agents=1000):
    comparaison = comparaisonMethodes(
        nb_agents=nb_agents,
        k=horizon,
        machines=BANDITS,
        eps=eps,
        c=c,
        initial_value=initial_value,
    )
    comparaison.run()
    comparaison.plot_comparaisons_mean_reward(
        title="Mean reward for different methods", figname="comparaison_mean_reward.jpg"
    )
    comparaison.plot_comparaisons_mean_reward(
        title="Accuracy for different methods", figname="comparaison_best_option.jpg"
    )


comparaison()
etude_unique_epsilon()
etude_moyenne_agents_epsilon()
etude_unique_optimistic()
etude_moyenne_agents_optimistic()
etude_unique_ucb()
etude_moyenne_agents_ucb()
