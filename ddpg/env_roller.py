import collections
import numpy as np

from ddpg.deque_backend import DequeBufferBackend
from local.openai.baselines.common.running_mean_std import RunningMeanStd


class EnvironmentRoller:
    def __init__(self, model, environment, buffer_capacity=1_000_000, buffer_initial_size=2_000, action_noise=None,
                 normalize_returns=False, normalize_observations=False, discount_factor=0.99):
        self.model = model
        self.environment = environment

        self.last_observation = self.environment.reset()
        self.action_noise = action_noise

        self.buffer_initial_size = buffer_initial_size
        self.buffer_capacity = buffer_capacity

        self.buffer = DequeBufferBackend(
            buffer_capacity=buffer_capacity,
            observation_space=environment.observation_space,
            action_space=environment.action_space
        )

        self.total_frames = 0
        self.discount_factor = discount_factor

        self.ob_rms = RunningMeanStd(shape=self.environment.observation_space.shape) if normalize_observations else None
        self.ret_rms = RunningMeanStd(shape=()) if normalize_returns else None
        self.clip_obs = 10.0
        self.accumulated_return = 0.0

        self.episode_rewards = collections.deque(maxlen=100)
        self.episode_lengths = collections.deque(maxlen=100)

    def roll_out(self):
        """ Evaluate environment for a single step and store in the buffer """
        action = self.model.action(self._filter_observation(self.last_observation[None]))[0]

        if self.action_noise is not None:
            noise = self.action_noise()

            action = np.clip(
                action + noise, self.environment.action_space.low, self.environment.action_space.high
            )

        new_obs, reward, done, info = self.environment.step(action)

        if self.ob_rms is not None:
            self.ob_rms.update(new_obs[None])

        if self.ret_rms is not None:
            self.accumulated_return = reward + self.discount_factor * self.accumulated_return

            self.ret_rms.update(np.array([self.accumulated_return]))

        self.buffer.store_transition(self.last_observation, action, reward, done)
        self.total_frames += 1

        # Usual, reset on done
        if done:
            new_obs = self.environment.reset()

            if self.action_noise is not None:
                self.action_noise.reset()

            self.accumulated_return = 0.0

        if 'episode' in info:
            self.episode_rewards.append(info['episode']['r'])
            self.episode_lengths.append(info['episode']['l'])

        self.last_observation = new_obs

        return {
        }

    def _filter_observation(self, obs):
        """ Potentially normalize observation """
        if self.ob_rms is not None:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8), -self.clip_obs, self.clip_obs)

            return obs.astype(np.float32)
        else:
            return obs

    def average_episode_reward(self):
        if self.episode_rewards:
            return np.mean(self.episode_rewards)
        else:
            return 0.0

    def average_episode_length(self):
        if self.episode_rewards:
            return np.mean(self.episode_lengths)
        else:
            return 0.0

    def is_ready_for_sampling(self):
        """ Return true if buffer has enough elements to perform sampling"""
        return self.buffer.current_size >= self.buffer_initial_size

    def sample_batch(self, batch_size):
        """ Return a random batch from replay buffer"""
        indexes = self.buffer.sample_batch_uniform(batch_size, history_length=1)
        batch = self.buffer.get_batch(indexes, history_length=1)

        rewards = batch['rewards'].astype(np.float32)

        if self.ret_rms is not None:
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + 1e-8), -self.clip_obs, self.clip_obs)

        return {
            'observations': self._filter_observation(batch['states']),
            'observations_next': self._filter_observation(batch['states+1']),
            'rewards': rewards,
            'dones': batch['dones'].astype(np.float32),
            'actions': batch['actions']
        }
