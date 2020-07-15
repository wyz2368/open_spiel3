from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections

import numpy as np
import ray
from open_spiel.python import rl_agent

import open_spiel.python.algorithms.psro_v2.ars_ray.optimizers as optimizers
import open_spiel.python.algorithms.psro_v2.ars_ray.utils as utils
from open_spiel.python.algorithms.psro_v2.ars_ray.shared_noise import *

"""
This is a customized parallel implementation of augmented random search (ARS) for openspiel.
https://arxiv.org/abs/1803.07055

The code is adapted from https://github.com/modestyachts/ARS/blob/master
The style of the code follows policy_gradient.py.

This version corresponds to ARS-V1 without observation normalization.
"""

Transition = collections.namedtuple(
    "Transition", "info_state action reward discount legal_actions_mask")

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


class ARS(rl_agent.AbstractAgent):
    """
    Main class for ARS. Now ARS only support discrete actions.
    This is version compatible with a parallel implementation.
    """
    def __init__(self,
                 session,
                 player_id,
                 info_state_size,
                 num_actions,
                 learning_rate=0.02,
                 nb_directions=160,
                 nb_best_directions=160,
                 noise=0.03
                 ):

        """
        Initialize the ARS agent.
        :param session: A dummy API place holder.
        :param player_id: int, player identifier. Usually its position in the game.
        :param info_state_size: int, info_state vector size.
        :param num_actions: int, number of actions per info state.
        :param learning_rate: Learning rate for ars.
        :param nb_directions: Number of Gaussian noise sampled.
        :param nb_best_directions: Number of exploration directions with the best performance.
        :param noise: Noise coefficient.
        """

        super(ARS, self).__init__(player_id)
        self._kwargs = locals()
        self._kwargs.pop("self")
        self._kwargs.pop("__class__")

        assert nb_best_directions <= nb_directions

        self.player_id = player_id
        self._info_state_size = info_state_size
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._nb_directions = nb_directions
        self._nb_best_directions = nb_best_directions
        self._noise = noise

        # Initialize the policy.
        self.theta = np.zeros((self._num_actions, self._info_state_size))

        # Initialize optimizer.
        self.optimizer = optimizers.SGD(self.theta, self._learning_rate)

        self._deterministic_policy = True


    def _act(self, info_state, legal_actions, is_evaluation, noise=None):
        # Make a singleton batch vector for ARS.
        info_state = np.reshape(info_state, [-1, 1])

        if is_evaluation or noise is None:
            policy_probs = softmax(self.theta.dot(info_state)).reshape(-1)
        else:
            cur_policy = self.theta + noise
            policy_probs = softmax(cur_policy.dot(info_state)).reshape(-1)

        # Remove illegal actions, re-normalize probs
        probs = np.zeros(self._num_actions)
        probs[legal_actions] = policy_probs[legal_actions]
        if sum(probs) != 0:
            probs /= sum(probs)
        else:
            probs[legal_actions] = 1 / len(legal_actions)

        if self._deterministic_policy:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p=probs)

        return action, probs

    def step(self, time_step, is_evaluation=False, noise=None):
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player()):
            # info_state has shape (dim,).
            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]
            action, probs = self._act(info_state, legal_actions, is_evaluation, noise)
        else:
            action = None
            probs = []

        return rl_agent.StepOutput(action=action, probs=probs)

    def _pi_update(self, rollout_rewards, deltas_idx):
        """
        Update current policy by rewards collected from different directions.
        :param rollout_rewards: [[pos rew1, neg rew1],
                                 [pos rew2, neg rew2]]
        :param deltas_idx: indices of noise used.
        """
        max_rewards = np.max(rollout_rewards, axis=1)

        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100 * (1 - (self._nb_best_directions / self._nb_directions)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                  (self.deltas.get(idx, self.theta.size)
                                                   for idx in deltas_idx),
                                                  batch_size=500)

        g_hat /= deltas_idx.size

        self.theta -= self.optimizer._compute_step(g_hat).reshape(self.theta.shape)

    def get_weights(self):
        return self.theta

    def set_weights(self, variables):
        self.theta = variables


    def copy_with_noise(self, sigma=0.0, copy_weights=True):
        """
        Copies the object and perturbates its network's weights with noise.
        :param sigma:
        :param copy_weights:
        :return:
        """

        copied_object = ARS(**self._kwargs)

        if copy_weights:
            copied_object.theta = self.theta.copy()

        copied_object.theta += sigma * np.random.normal(size=np.shape(self.theta))

        return copied_object


