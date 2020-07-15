import pyspiel
import tensorflow.compat.v1 as tf

import concurrent.futures
from open_spiel.python import rl_environment

from open_spiel.python.algorithms.psro_v2.parallel.worker import do_something


sess = tf.Session()

with concurrent.futures.ProcessPoolExecutor() as executor:
    secs = ['kuhn_poker', 'leduc_poker']
    results = executor.map(do_something, secs)

    for result in results:
        print(result)