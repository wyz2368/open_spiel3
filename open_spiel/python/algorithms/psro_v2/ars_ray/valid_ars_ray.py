import pyspiel
import sys
import os

import logging
logging.disable(logging.INFO)
import functools
print = functools.partial(print, flush=True)


from open_spiel.python import rl_environment
# from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy

import ray
import cloudpickle

from open_spiel.python.algorithms.psro_v2.ars_ray.workers import worker
from open_spiel.python.algorithms.psro_v2.parallel.worker import do_something
import concurrent.futures



# redis_password = sys.argv[1]
# num_cpus = int(sys.argv[2])

game = pyspiel.load_game_as_turn_based("kuhn_poker",
                                      {"players": pyspiel.GameParameter(
                                             2)})
env = rl_environment.Environment(game)
env.reset()

sess = None

info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
# print(info_state_size, num_actions)
agent_class = rl_policy.ARSPolicy_parallel
agent_kwargs = {
    "session": sess,
    "info_state_size": info_state_size,
    "num_actions": num_actions,
    "learning_rate": 0.03,
    "nb_directions": 32,
    "nb_best_directions": 32,
    "noise": 0.07
}

# oracle = rl_oracle.RLOracle(
#     env,
#     agent_class,
#     agent_kwargs,
#     number_training_episodes=1000,
#     self_play_proportion=0.0,
#     sigma=0.0,
#     num_workers=4,
#     ars_parallel=True
# )



# ray.init(address=os.environ["ip_head"], redis_password=redis_password)
ray.init(temp_dir='./ars_temp_dir/')
for _ in range(2):
    print(ray.get(worker.remote("kuhn_poker")))



# agents = [
#     agent_class(
#       env,
#       player_id,
#       **agent_kwargs)
#     for player_id in range(2)
#   ]



print("Done")
ray.shutdown()