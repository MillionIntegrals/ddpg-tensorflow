import os.path

import sys
import numpy as np
import pandas as pd
import time
import random
import gym
import tensorflow as tf
import tqdm
import shutil

import local.openai.baselines.logger as logger

from ddpg.algo import DeepDeterministicPolicyGradient
from local.openai.baselines.bench import Monitor
from local.env_normalize import EnvNormalize

from ddpg.model import DeterministicPolicyModel
from ddpg.env_roller import EnvironmentRoller
from ddpg.action_noise import OrnsteinUhlenbeckNoiseProcess


def make_tf_session():
    """ Create a simple tensorflow session """
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    # config.log_device_placement = True

    session = tf.Session(config=config)
    return session


def build_environment(envname, logdir, seed, normalize_observations=False, normalize_returns=False,
                      allow_early_resets=False, normalize_gamma=0.99):
    """ Construct environment for specified problem """
    env = gym.make(envname)
    env.seed(seed)

    os.makedirs(logdir, exist_ok=True)

    env = Monitor(env, logdir, allow_early_resets=allow_early_resets)

    if normalize_observations or normalize_returns:
        env = EnvNormalize(
            env,
            normalize_observations=normalize_observations,
            normalize_returns=normalize_returns,
            gamma=normalize_gamma
        )

    return env


def train_ddpg_loop(
        run_name,
        seed,
        discount_factor=0.99,

        total_frames=1_000_000,

        steps_per_epoch=20,
        epoch_rollout_steps=100,
        epoch_training_steps=50,

        env_normalize_observations=False,
        env_normalize_returns=False,

        roller_normalize_observations=False,
        roller_normalize_returns=False,

        batch_size=64,

        tau=0.01,
        actor_lr=1e-3,
        critic_lr=1e-4,
        critic_l2_reg=0.01,

        ops_together=False,

        envname="HalfCheetah-v2",
        device="/device:GPU:0"
):
    session = make_tf_session()

    start_time = time.time()

    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    logdir = os.path.join('.', 'log', run_name)
    tb_dir = os.path.join('.', 'tensorboard', run_name)

    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)

    logger.configure(dir=logdir)

    environment = build_environment(
        envname, logdir, seed,
        normalize_observations=env_normalize_observations,
        normalize_returns=env_normalize_returns,
        normalize_gamma=discount_factor
    )

    with tf.device(device):
        model = DeterministicPolicyModel(
            "model", session, environment.observation_space, environment.action_space
        )

        target_model = DeterministicPolicyModel(
            "target", session, environment.observation_space, environment.action_space
        )

        algo = DeepDeterministicPolicyGradient(
            "ddpg", session, model, target_model,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            critic_l2_reg=critic_l2_reg,
            discount_factor=discount_factor,
            ops_together=ops_together
        )

    len_action_space = environment.action_space.shape[0]

    action_noise = OrnsteinUhlenbeckNoiseProcess(
        np.zeros(len_action_space), 0.2 * np.ones(len_action_space)
    )

    env_roller = EnvironmentRoller(
        model=model,
        environment=environment,
        action_noise=action_noise,
        normalize_returns=roller_normalize_returns,
        normalize_observations=roller_normalize_observations,
        discount_factor=discount_factor
    )

    summary_writer = tf.summary.FileWriter(logdir=tb_dir, graph=session.graph)

    algo.initialize_training()

    num_epochs = total_frames // (steps_per_epoch * epoch_rollout_steps)

    # Train model
    for epoch_idx in range(1, num_epochs+1):
        train_metrics_aggregator = []
        rollout_metrics_aggregator = []

        for _ in tqdm.trange(steps_per_epoch, file=sys.stdout):
            if not env_roller.is_ready_for_sampling():
                while not env_roller.is_ready_for_sampling():
                    rollout_metrics_aggregator.append(env_roller.roll_out())
            else:
                for rollout_step in range(epoch_rollout_steps):
                    rollout_metrics_aggregator.append(env_roller.roll_out())

            for train_step in range(epoch_training_steps):
                rollout = env_roller.sample_batch(batch_size)

                train_metrics_aggregator.append(algo.train(rollout))

        train_metrics = pd.DataFrame(train_metrics_aggregator).mean().to_dict()
        rollout_metrics = pd.DataFrame(rollout_metrics_aggregator).mean().to_dict()

        current_time = time.time()
        elapsed_time = (current_time - start_time)

        epoch_metrics_dict = {
            'epoch_idx': epoch_idx,
            'elapsed_time': elapsed_time / 60 / 60,  # In hours
            'total_frames': env_roller.total_frames,
            'fps': int(env_roller.total_frames / elapsed_time),
            'episode_rewards': env_roller.average_episode_reward(),
            'episode_lengths': env_roller.average_episode_length()
        }

        for item, value in train_metrics.items():
            epoch_metrics_dict[item] = value

        for item, value in rollout_metrics.items():
            epoch_metrics_dict[item] = value

        summary = tf.Summary()

        for item, value in sorted(epoch_metrics_dict.items()):
            logger.record_tabular(item, value)
            summary.value.add(tag=item, simple_value=value)

        logger.dump_tabular()
        summary_writer.add_summary(summary, global_step=epoch_idx)
