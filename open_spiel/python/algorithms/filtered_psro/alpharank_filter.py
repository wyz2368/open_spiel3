import numpy as np

def alpharank_filter(meta_games,
                     policies,
                     marginals,
                     size_threshold=20,
                     keep_dim=True):
    """
    Use alpharank to filter out the transient strategies in the empirical game.
    :param meta_games: PSRO meta_games
    :param policies: a list of list of strategies, one per player.
    :param marginals: a list of list of marginal alpharank distribution, one per player.
    :param size_threshold: maximum size for each meta game. (unused)
    :param keep_dim: keep all players having the same number of strategies.
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
        # if keep_dim:
        #     min_num_str_filtered = 1000
        #     for idx in filtered_idx_list:
        #         min_num_str_filtered = min(len(idx), min_num_str_filtered)
        #     for i, idx in enumerate(filtered_idx_list):
        #         if len(idx) > min_num_str_filtered:
        #             masses_strategy_pairs = sorted(zip(marginals[i][idx], idx))[:min_num_str_filtered]
        #             filtered_idx_list[i] = np.array(sorted([pair[1] for pair in masses_strategy_pairs]))

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
                                            marginals)
    print("meta_games:", meta_games)
    print("policies:", policies)


# alpharank_filter_test()