import numpy as np
import gym.spaces as spaces

import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.initializers as initializers


class DeterministicPolicyModel:
    """ DDPG model """

    def __init__(self, name, session, observation_space, action_space):
        self.name = name

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        assert (np.abs(action_space.low) == action_space.high).all()  # we assume symmetric actions.

        self.session = session

        with tf.variable_scope(self.name):
            self.observations_input = tf.placeholder(
                observation_space.dtype,
                shape=tuple([None] + list(observation_space.shape)),
                name="observations_input"
            )

            self.actions_input = tf.placeholder(
                action_space.dtype,
                shape=tuple([None] + list(action_space.shape)),
                name="actions_input"
            )

            self.actions_size = action_space.shape[0]

            self.combined_input = tf.concat([self.observations_input, self.actions_input], axis=1, name="combined_input")

            with tf.variable_scope("actor"):
                self.actor01 = layers.dense(
                    inputs=self.observations_input, units=64, activation=tf.tanh, name="layer_one",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.orthogonal(gain=np.sqrt(2))
                )

                self.actor02 = layers.dense(
                    inputs=self.actor01, units=64, activation=tf.tanh, name="layer_two",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.orthogonal(gain=np.sqrt(2))
                )

                self.actor_head = layers.dense(
                    inputs=self.actor02, units=self.actions_input.shape[1].value, activation=tf.tanh, name="head",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.orthogonal(gain=0.01)
                )

                self.actor_head_rescaled = self.actor_head * action_space.high

            self.model_computed_combined = tf.concat(
                [self.observations_input, self.actor_head], axis=1, name="model_computed_combined"
            )

            with tf.variable_scope("critic"):
                self.critic01 = layers.dense(
                    inputs=self.combined_input, units=64, activation=tf.tanh, name="layer_one",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.orthogonal(gain=np.sqrt(2))
                )

                self.critic02 = layers.dense(
                    inputs=self.critic01, units=64, activation=tf.tanh, name="layer_two",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.orthogonal(gain=np.sqrt(2))
                )

                self.action_value_head = layers.dense(
                    inputs=self.critic02, units=1, activation=None, name="head",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.random_uniform(-3.0e-3, 3.0e-3)
                )

            with tf.variable_scope("critic"):
                self.model_critic01 = layers.dense(
                    inputs=self.model_computed_combined, units=64, activation=tf.tanh, name="layer_one",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.orthogonal(gain=np.sqrt(2)),
                    reuse=True  # Use the same weights for 'critic' and for 'model critic'
                )

                self.model_critic02 = layers.dense(
                    inputs=self.model_critic01, units=64, activation=tf.tanh, name="layer_two",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.orthogonal(gain=np.sqrt(2)),
                    reuse=True  # Use the same weights for 'critic' and for 'model critic'
                )

                self.state_value_head = layers.dense(
                    inputs=self.model_critic02, units=1, activation=None, name="head",
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.random_uniform(-3.0e-3, 3.0e-3),
                    reuse=True  # Use the same weights for 'critic' and for 'model critic'
                )

    def variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def actor_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}/{}'.format(self.name, "actor"))

    def actor_trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/{}'.format(self.name, "actor"))

    def critic_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}/{}'.format(self.name, "critic"))

    def critic_trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/{}'.format(self.name, "critic"))

    def action(self, observations):
        """ Calculate action """
        return self.session.run(self.actor_head_rescaled, feed_dict={self.observations_input: observations})

    def state_value(self, observations):
        """ Calculate state value """
        return self.session.run(self.state_value_head, feed_dict={self.observations_input: observations})

