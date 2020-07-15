import numpy as np
from open_spiel.python.algorithms.nash_solver.general_nash_solver import nash_solver
import itertools
from open_spiel.python.algorithms.psro_v2 import meta_strategies
import pickle
import os

def regret(meta_games, subgame_index, subgame_ne=None):
    """
    Calculate the regret based on a complete payoff matrix for PSRO
    Assume all players have the same number of policies
    :param meta_games: meta_games in PSRO
    :param subgame_index: the subgame to evaluate. Redundant when subgame_ne is supplied
    :param: subgame_ne: subgame nash equilibrium vector.
    :return: a list of regret, one for each player.
    """
    num_policy = np.shape(meta_games[0])[0]
    num_players = len(meta_games)
    if num_policy == subgame_index:
        print("The subgame is same as the full game. Return zero regret.")
        return np.zeros(num_players)
    num_new_pol = num_policy - subgame_index

    index = [list(np.arange(subgame_index)) for _ in range(num_players)]
    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]
    nash = nash_solver(submeta_games, solver="gambit") if not subgame_ne else subgame_ne
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(nash)
    this_meta_prob = [np.append(nash[i],[0 for _ in range(num_new_pol)]) for i in range(num_players)]

    nash_payoffs = []
    deviation_payoffs = []

    for i in range(num_players): 
        ne_payoff = np.sum(submeta_games[i]*prob_matrix)
        # iterate through player's new policy
        dev_payoff = []
        for j in range(num_new_pol):
            dev_prob = this_meta_prob.copy()
            dev_prob[i] = np.zeros(num_policy)
            dev_prob[i][subgame_index+j] = 1
            new_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
            dev_payoff.append(np.sum(meta_games[i]*new_prob_matrix))
        deviation_payoffs.append(dev_payoff-ne_payoff)
        nash_payoffs.append(ne_payoff)
    
    regret = np.maximum(np.max(deviation_payoffs,axis=1),0)
    return regret

def strategy_regret(meta_games, subgame_index, ne=None, subgame_ne=None):
    """
        Calculate the strategy regret based on a complete payoff matrix for PSRO.
        strategy_regret of player equals to nash_payoff in meta_game - fix opponent nash strategy, player deviates to subgame_nash
        Assume all players have the same number of policies.
        :param meta_games: meta_games in PSRO
        :param subgame_index: subgame to evaluate, redundant if subgame_nash supplied
        :param: nash: equilibrium vector
        :param: subgame_ne: equilibrium vector
        :return: a list of regret, one for each player.

    """
    num_players = len(meta_games)
    num_new_pol = np.shape(meta_games[0])[0] - subgame_index

    ne = nash_solver(meta_games, solver="gambit") if not ne else ne
    index = [list(np.arange(subgame_index)) for _ in range(num_players)]
    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]
    subgame_ne = nash_solver(submeta_games, solver="gambit") if not subgame_ne else subgame_ne
    nash_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(ne)

    regrets = []
    for i in range(num_players):
        ne_payoff = np.sum(meta_games[i]*nash_prob_matrix)
        dev_prob = ne.copy()
        dev_prob[i] = list(np.append(subgame_ne[i],[0 for _ in range(num_new_pol)]))
        dev_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
        subgame_payoff = np.sum(meta_games[i]*dev_prob_matrix)
        regrets.append(ne_payoff-subgame_payoff)

    return regrets

def sample_episode(env, agents):
    """
    sample pure strategy payoff in an env
    Params:
        agents : a list of length num_player
        env    : open_spiel environment
    Returns:
        a list of length num_player containing players' strategies
    """
    time_step = env.reset()
    cumulative_rewards = 0.0
    while not time_step.last():
      if time_step.is_simultaneous_move():
        action_list = []
        for agent in agents:
          output = agent.step(time_step, is_evaluation=True)
          action_list.append(output.action)
        time_step = env.step(action_list)
        cumulative_rewards += np.array(time_step.rewards)
      else:
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step, is_evaluation=False)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        cumulative_rewards += np.array(time_step.rewards)

    return cumulative_rewards

def rollout(env, strategies, strategy_support, sims_per_entry=1000):
    """
    Evaluate player's mixed strategy with support in env.
    Params:
        env              : an open_spiel env
        strategies       : list of list, each list containing a player's strategy
        strategy_support : mixed_strategy support probability vector
        sims_per_entry   : number of episodes for each pure strategy profile to sample
    Return:
        a list of players' payoff
    """
    num_players = len(strategies)
    num_strategies = [len(ele) for ele in strategies]
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(strategy_support)
    payoff_tensor = np.zeros([num_players]+num_strategies)

    for ind in itertools.product(*[np.arange(ele) for ele in num_strategies]):
        strat = [strategies[i][ind[i]] for i in range(num_players)]
        pure_payoff = np.zeros(num_players)
        for _ in range(sims_per_entry):
            pure_payoff += sample_episode(env, strat)
        payoff_tensor[tuple([...]+list(ind))] = pure_payoff/sims_per_entry

    return [np.sum(payoff_tensor[i]*prob_matrix) for i in range(num_players)]


class SElogs(object):
    def __init__(self,
                 slow_oracle_period,
                 fast_oracle_period,
                 meta_strategy_methods,
                 heuristic_list):

        self.slow_oracle_period = slow_oracle_period
        self.fast_oracle_period = fast_oracle_period
        self.meta_strategy_methods = meta_strategy_methods
        self.heuristic_list = heuristic_list

        self._slow_oracle_iters = []
        self._fast_oracle_iters = []

        self.regrets = []
        self.nashconv = []


    def update_regrets(self, regrets):
        self.regrets.append(regrets)

    def get_regrets(self):
        return self.regrets

    def update_nashconv(self, nashconv):
        self.nashconv.append(nashconv)

    def get_nashconv(self):
        return self.nashconv

    def update_slow_iters(self, iter):
        self._slow_oracle_iters.append(iter)

    def get_slow_iters(self):
        return self._slow_oracle_iters

    def update_fast_iters(self, iter):
        self._fast_oracle_iters.append(iter)

    def get_fast_iters(self):
        return self._fast_oracle_iters


def smoothing_kl(p, q, eps=0.001):
    p = smoothing(p, eps)
    q = smoothing(q, eps)
    return np.sum(p * np.log(p / q))


def smoothing(p, eps):
    p = np.array(p, dtype=np.float)
    zeros_pos_p = np.where(p == 0)[0]
    num_zeros = len(zeros_pos_p)
    x = eps * num_zeros / (len(p) - num_zeros)
    for i in range(len(p)):
        if i in zeros_pos_p:
            p[i] = eps
        else:
            p[i] -= x
    return p


def isExist(path):
    """
    Check if a path exists.
    :param path: path to check.
    :return: bool
    """
    return os.path.exists(path)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if isExists:
        raise ValueError(path + " already exists.")
    else:
        os.makedirs(path)
        print(path + " has been created successfully.")

def save_pkl(obj,path):
    """
    Pickle a object to path.
    :param obj: object to be pickled.
    :param path: path to save the object
    """
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    """
    Load a pickled object from path
    :param path: path to the pickled object.
    :return: object
    """
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path,'rb') as f:
        result = pickle.load(f)
    return result


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def save_strategies(solver, checkpoint_dir):
    """
    Save all strategies.
    """
    num_players = solver._num_players
    for player in range(num_players):
        current_path = os.path.join(checkpoint_dir, 'strategies/player_' + str(player) + "/")
        if not isExist(current_path):
            mkdir(current_path)
        for i, policy in enumerate(solver.get_policies()[player]):
            if isExist(current_path + str(i+1) + '.pkl'):
                continue
            save_pkl(policy.get_weights(), current_path + str(i+1) + '.pkl')

