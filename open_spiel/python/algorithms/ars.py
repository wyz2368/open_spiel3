from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections

import numpy as np

from open_spiel.python import rl_agent

"""
This is a customized implementation of augmented random search (ARS) for openspiel.
https://arxiv.org/abs/1803.07055

The code is adapted from https://github.com/sourcecode369/Augmented-Random-Search-.
The style of the code follows policy_gradient.py.
"""

Transition = collections.namedtuple(
    "Transition", "info_state action reward discount legal_actions_mask")

# Normalizing the states

class Normalizer():
    """
    The normalizer normalizes the observations in ARS. Refer to the ARS-V2.
    """
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


class ARS(rl_agent.AbstractAgent):
    """
    Main class for ARS. Now ARS only support discrete actions.
    """
    def __init__(self,
                 session,
                 player_id,
                 info_state_size,
                 num_actions,
                 episode_length=1000,
                 learning_rate=0.02,
                 nb_directions=16,
                 nb_best_directions=16,
                 noise=0.03,
                 seed=123,
                 additional_discount_factor=1.0,
                 v2=False,
                 discrete_action=True,
                 deterministic=False):
        """
        Initialize the ARS agent.
        :param session: A dummy API place holder.
        :param player_id: int, player identifier. Usually its position in the game.
        :param info_state_size: int, info_state vector size.
        :param num_actions: int, number of actions per info state.
        :param episode_length: The maximum length of an episode.
        :param learning_rate: Learning rate for ars.
        :param nb_directions: Number of Gaussian noise sampled.
        :param nb_best_directions: Number of exploration directions with the best performance.
        :param noise: Noise coefficient.
        :param seed: Random seed.
        :param additional_discount_factor: Additional discount factor for episodes.
        :param v2: bool, True: enable ARS-V2
        :param discrete_action: bool, True for problems with discrete action space.
        :param deterministic: bool, ars act deterministically or not
        """

        super(ARS, self).__init__(player_id)
        self._kwargs = locals()
        self._kwargs.pop("self")
        self._kwargs.pop("__class__")

        assert nb_best_directions <= nb_directions

        self.player_id = player_id
        self._info_state_size = info_state_size
        self._num_actions = num_actions
        self._episode_length = episode_length
        self._learning_rate = learning_rate
        self._nb_directions = nb_directions
        self._nb_best_directions = nb_best_directions
        self._noise = noise
        self._seed = seed
        self._extra_discount = additional_discount_factor
        self.v2 = v2
        self.discrete_action = discrete_action
        self.deterministic = deterministic

        if self.v2:
            self.normalizer = Normalizer(self._info_state_size)

        self._episode_data = []
        self._dataset = collections.defaultdict(list)
        self._prev_time_step = None
        self._prev_action = None

        # The index of current policy.
        self._current_policy_idx = -1

        # Initialize the policy.
        self.theta = np.zeros((self._num_actions, self._info_state_size))
        self.sample_deltas()
        self.deltas_iterator()

    def _act(self, info_state, legal_actions, is_evaluation):
        if self.v2:
            self.normalizer.observe(info_state)
            info_state = self.normalizer.normalize(info_state)

        # Make a singleton batch vector for ARS.
        info_state = np.reshape(info_state, [-1, 1])
        if self.discrete_action:
            if is_evaluation:
                policy_probs = softmax(self.theta.dot(info_state)).reshape(-1)
            else:
                policy_probs = softmax(self._policy.dot(info_state)).reshape(-1)
        else:
            raise NotImplementedError("The ARS currently does not support continuous actions.")

        # Remove illegal actions, re-normalize probs
        probs = np.zeros(self._num_actions)
        probs[legal_actions] = policy_probs[legal_actions]
        if sum(probs) != 0:
            probs /= sum(probs)
        else:
            probs[legal_actions] = 1 / len(legal_actions)

        if not self.deterministic:
          action = np.random.choice(len(probs), p=probs)
        else:
          action = np.argmax(probs)
        return action, probs

    def step(self, time_step, is_evaluation=False):
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player()):
            # info_state has shape (dim,).
            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]
            action, probs = self._act(info_state, legal_actions, is_evaluation)
            
        else:
            action = None
            probs = []

        if not is_evaluation:
            # Add data points to current episode buffer.
            if self._prev_time_step:
                self._add_transition(time_step)

            # Episode done, add to dataset and maybe learn.
            if time_step.last():
                self._add_episode_data_to_dataset()
                
                direction = self._current_policy_idx // self._nb_directions
                delta_idx = self._current_policy_idx % self._nb_directions
                if direction == 0:
                    self._pos_rew[delta_idx] = self._dataset["returns"]
                    self._dataset = collections.defaultdict(list)
                elif direction == 1:
                    self._neg_rew[delta_idx] = self._dataset["returns"]
                    self._dataset = collections.defaultdict(list)
                else:
                    raise ValueError("Number of directions tried beyond scope.")

                self.deltas_iterator()
                self._prev_time_step = None
                self._prev_action = None
                return
            else:
                self._prev_time_step = time_step
                self._prev_action = action

        return rl_agent.StepOutput(action=action, probs=probs)

    def sample_deltas(self):
        """
        Sample self._nb_directions number of Gausian noise, each of which matches the shape of theta.
        :return:
        """
        self._deltas = [np.random.randn(*self.theta.shape) for _ in range(self._nb_directions)]
        self._pos_rew = [None] * self._nb_directions
        self._neg_rew = [None] * self._nb_directions
        self._deltas_idx = 0

    def deltas_iterator(self):
        """
        Generate noisy policy based on sampled noise.
        There is an outlying case, it does not really take place, but just exist to facilitates code writing: 
            self._current_policy_idx = 128 
            self._deltas_idx = 129
        This case will be detected by self.done and pi_update will be consequently called to reset the indexes. The "self._current_policy_idx == 2*self._nb_directions" in step(self) is also designed for this purpose
        """
        direction = self._deltas_idx // self._nb_directions
        if direction == 0:
            sign = 1
        elif direction == 1:
            sign = -1
        elif direction == 2:
            # If all noisy polies have been tried then update policy.
            # Not update policy at the end of evey episode in PSRO.
            self._pi_update()
            self.sample_deltas()
            self.deltas_iterator()
            return
        else:
            raise ValueError("Number of directions tried beyond scope.")
        delta_idx = self._deltas_idx % self._nb_directions
        self._policy = self.theta + sign * self._noise * self._deltas[delta_idx]
        self._current_policy_idx = self._deltas_idx
        self._deltas_idx += 1

    def _pi_update(self):
        """
        Update current policy by rewards collected from different directions.
        :return:
        """
        if None in self._pos_rew or None in self._neg_rew:
            raise ValueError("Not all directions are evaluated.")

        all_rewards = np.array(self._pos_rew + self._neg_rew)

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(self._pos_rew, self._neg_rew))}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self._nb_best_directions]
        rollouts = [(self._pos_rew[k], self._neg_rew[k], self._deltas[k]) for k in order]

        sigma_r_array = []
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
            sigma_r_array.extend([r_pos,r_neg])
        sigma_r = np.array(sigma_r_array).std()
        
        if sigma_r == 0:
          sigma_r = 1

        self.theta += self._learning_rate / (self._nb_best_directions * sigma_r) * step
        return sigma_r

    def get_weights(self):
        return self.theta

    def set_weights(self, variables):
        self.theta = variables

    def _add_episode_data_to_dataset(self):
        """Add episode data to the buffer."""
        info_states = [data.info_state for data in self._episode_data]
        rewards = [data.reward for data in self._episode_data]
        discount = [data.discount for data in self._episode_data]
        actions = [data.action for data in self._episode_data]

        # Calculate returns of an episode
        # final value equals to the first element of the list
        returns = np.array(rewards)
        for idx in reversed(range(len(rewards[:-1]))):
            returns[idx] = (
                    rewards[idx] +
                    discount[idx] * returns[idx + 1] * self._extra_discount)

        # Add flattened data points to dataset
        self._dataset["actions"].extend(actions)
        self._dataset["returns"] = returns[0]
        self._dataset["info_states"].extend(info_states)
        self._episode_data = []


    def _add_transition(self, time_step):
        """Adds intra-episode transition to the `_episode_data` buffer.

        Adds the transition from `self._prev_time_step` to `time_step`.

        Args:
          time_step: an instance of rl_environment.TimeStep.
        """
        assert self._prev_time_step is not None
        legal_actions = (
            self._prev_time_step.observations["legal_actions"][self.player_id])
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        transition = Transition(
            info_state=(
                self._prev_time_step.observations["info_state"][self.player_id][:]),
            action=self._prev_action,
            reward=time_step.rewards[self.player_id],
            discount=time_step.discounts[self.player_id],
            legal_actions_mask=legal_actions_mask)

        self._episode_data.append(transition)

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
