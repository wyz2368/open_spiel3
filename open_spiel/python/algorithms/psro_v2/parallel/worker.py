import pyspiel
from open_spiel.python import rl_environment


def do_something(game_name):
    game = pyspiel.load_game_as_turn_based(game_name,
                                                   {"players": pyspiel.GameParameter(
                                                       2)})
    env = rl_environment.Environment(game)
    return env.name