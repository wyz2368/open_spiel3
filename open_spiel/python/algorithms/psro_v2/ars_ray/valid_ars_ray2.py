import ray


@ray.remote
class Worker(object):
    def __init__(self,
                 env_name,
                 deltas=None,
                 slow_oracle_kargs=None,
                 fast_oracle_kargs=None
                 ):
        # initialize rl environment.

        import pyspiel

        self._env_name = env_name
        print(env_name)


class Run():
    def __init__(self):
        ray.init(temp_dir='./ars_temp_dir/')
        self._workers = [Worker.remote("kuhn") for _ in range(3)]


a = Run()