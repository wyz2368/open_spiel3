import numpy as np


def strategy_filter(solver):
    num_players = solver._num_players
    filtered_player = solver._filtered_player
    _meta_games = solver._meta_games
    _policies = solver._policies

    filtered_idx = len(_policies[filtered_player]) - 1

    for player in range(num_players):
        # filter meta_games.
        _meta_games[player] = np.delete(_meta_games[player], filtered_idx, axis=filtered_player)

    # filter policies.
    _policies[filtered_player] = np.delete(_policies[filtered_player], filtered_idx)
    _policies[filtered_player] = list(_policies[filtered_player])

    return _meta_games, _policies