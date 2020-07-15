import numpy as np

from open_spiel.python.algorithms.psro_v2.utils import alpharank_strategy

def strategy_filter(solver, stopping_time=None):
    """

    :param solver:
    :param stopping_time: time step when we stop filtering strategies.
    :return:
    """
    if stopping_time is not None:
        if solver._iterations > stopping_time:
            return solver._meta_games, solver._policies
    if solver.filtering_method == "alpharank":
        marginals, _ = alpharank_strategy(solver, return_joint=True)
        return alpharank_filter(solver._meta_games,
                                solver._policies,
                                marginals,
                                solver.strategy_set_size)
    elif solver.filtering_method == "etrace":
        return etrace_filter(solver)
    elif solver.filtering_method == "one_nash":
        return one_nash_filter(solver)
    else:
        return solver._meta_games, solver._policies


def alpharank_filter(meta_games,
                     policies,
                     marginals,
                     size_threshold):
    """
    Use alpharank to filter out the transient strategies in the empirical game.
    :param meta_games: PSRO meta_games
    :param policies: a list of list of strategies, one per player.
    :param marginals: a list of list of marginal alpharank distribution, one per player.
    :param size_threshold: maximum size for each meta game. (unused)
    :return:
    """
    # TODO:add skip functionality.
    num_str, _ = np.shape(meta_games[0])
    if num_str <= size_threshold:
        return meta_games, policies
    num_players = len(meta_games)
    filtered_idx_list = []
    for player in range(num_players):
        lowest_ranked_str = np.argmin(marginals[player])
        filtered_idx_list.append([lowest_ranked_str])

    for player in range(num_players):
        # filter meta_games.
        for dim in range(num_players):
            filtered_idx = filtered_idx_list[dim]
            meta_games[player] = np.delete(meta_games[player], filtered_idx, axis=dim)
        # filter policies.
        policies[player] = np.delete(policies[player], filtered_idx_list[player])
        policies[player] = list(policies[player])

    print("Strategies filtered:")
    num_str_players = []
    for player in range(num_players):
        print("Player " + str(player) + ":", filtered_idx_list[player])
        num_str_players.append(len(policies[player]))
    print("Number of strategies after filtering:", num_str_players)

    return meta_games, policies

def alpharank_filter_test():
    meta_game = np.array([[1,2,3,4,5],
                          [2,3,4,5,6],
                          [3,4,5,6,7],
                          [4,5,6,7,8],
                          [5,6,7,8,9]])
    meta_games = [meta_game, -meta_game]

    policies = [[1,2,3,4,5], [6,7,8,9,10]]
    marginals = [np.array([0.001, 0.3, 0.3, 0.3, 0.009]), np.array([0.3, 0.001, 0.001, 0.3, 0.001])]
    meta_games, policies = alpharank_filter(meta_games,
                                            policies,
                                            marginals,
                                            4)
    print("meta_games:", meta_games)
    print("policies:", policies)

def etrace_filter(solver, gamma=0.5, threshold=0.001):
    num_players = solver._num_players
    filtered_idx_list = []
    for player in range(num_players):
        solver.etrace[player] *= gamma
    solver.update_meta_strategies()
    nash = solver.get_meta_strategies()
    for player in range(num_players):
        one_pos = np.where(nash[player] > 0.005)[0]
        zero_pos = np.where(nash[player] <= 0.005)[0]
        nash[player][one_pos] = 1.0
        nash[player][zero_pos] = 0.0
        solver.etrace[player] = np.append(solver.etrace[player], 0.0)
        solver.etrace[player] += nash[player]
        solver.etrace[player][solver.etrace[player] < threshold] = 0
        filtered_idx_list.append(np.argmin(solver.etrace[player]))

    print("Eligibility Trace with gamma:", gamma)
    for player in range(num_players):
        print("Player " + str(player) + ":", solver.etrace[player])

    meta_games = solver.get_meta_game()
    policies = solver.get_policies()
    num_str, _ = np.shape(meta_games[0])
    if num_str <= solver.strategy_set_size:
        return meta_games, policies
    for player in range(num_players):
        # filter meta_games.
        for dim in range(num_players):
            filtered_idx = filtered_idx_list[dim]
            meta_games[player] = np.delete(meta_games[player], filtered_idx, axis=dim)
        # filter policies.
        policies[player] = np.delete(policies[player], filtered_idx_list[player])
        policies[player] = list(policies[player])
        solver.etrace[player] = np.delete(solver.etrace[player], filtered_idx_list[player])

    print("Strategies filtered:")
    num_str_players = []
    for player in range(num_players):
        print("Player " + str(player) + ":", filtered_idx_list[player])
        num_str_players.append(len(policies[player]))
    print("Number of strategies after filtering:", num_str_players)

    return meta_games, policies

def one_nash_filter(solver, threshold=0.001):
    num_players = solver._num_players
    filtered_idx_list = []
    len_filtered_strategies = []

    meta_games = solver.get_meta_game()
    policies = solver.get_policies()
    num_str, _ = np.shape(meta_games[0])
    if num_str <= solver.strategy_set_size:
        return meta_games, policies

    solver.update_meta_strategies()
    nash = solver.get_meta_strategies()

    for player in range(num_players):
        zero_pos = np.where(nash[player] <= threshold)[0]
        filtered_idx_list.append(zero_pos)
        len_filtered_strategies.append(len(zero_pos))

    if np.sum(len_filtered_strategies) == 0:
        return solver._meta_games, solver._policies

    for player in range(num_players):
        # filter meta_games.
        for dim in range(num_players):
            filtered_idx = filtered_idx_list[dim]
            meta_games[player] = np.delete(meta_games[player], filtered_idx, axis=dim)
        # filter policies.
        policies[player] = np.delete(policies[player], filtered_idx_list[player])
        policies[player] = list(policies[player])

    print("Strategies filtered:")
    num_str_players = []
    for player in range(num_players):
        print("Player " + str(player) + ":", filtered_idx_list[player])
        num_str_players.append(len(policies[player]))
    print("Number of strategies after filtering:", num_str_players)

    return meta_games, policies

