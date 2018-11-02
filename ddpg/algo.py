import tensorflow as tf
import tensorflow.contrib as tc


class DeepDeterministicPolicyGradient:
    def __init__(self, name, session, model, target_model, tau=0.01, actor_lr=1e-4, critic_lr=1e-3,
                 critic_l2_reg=0.0,
                 discount_factor=0.99,
                 ops_together=False):
        self.name = name
        self.session = session
        self.model = model
        self.target_model = target_model
        self.tau = tau
        self.discount_factor = discount_factor
        self.ops_together = ops_together

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.critic_l2_reg = critic_l2_reg

        with tf.variable_scope(self.name):
            initialization_update_ops = []
            step_update_ops = []

            # Set up tensorflow ops to assign model vars to target vars
            for target_var, model_var in zip(self.target_model.variables(), self.model.variables()):
                initialization_update_ops.append(tf.assign(target_var, model_var))
                step_update_ops.append(tf.assign(target_var, target_var * (1.0 - self.tau) + model_var * self.tau))

            self.initialization_update_ops = tf.group(initialization_update_ops)
            self.step_update_ops = tf.group(step_update_ops)

            self.value_target_input = tf.placeholder(tf.float32, shape=(None, 1), name="target_value")

            self.critic_loss = tf.losses.mean_squared_error(
                self.value_target_input, self.model.action_value_head, scope="critic_loss"
            )
            self.actor_loss = -tf.reduce_mean(self.model.state_value_head, name="actor_loss")

            if self.critic_l2_reg > 0:
                critic_regularizer = tc.layers.apply_regularization(
                    regularizer=tc.layers.l2_regularizer(critic_l2_reg),
                    # That's how OpenAI are doing this...
                    weights_list=[x for x in self.model.critic_trainable_vars() if 'kernel' in x.name][-1:]
                )

                self.critic_loss += critic_regularizer

            # Set up optimizers
            self.critic_optimizer = tf.train.AdamOptimizer(
                learning_rate=critic_lr,
                epsilon=1e-8,
                name="critic_adam"
            )

            self.critic_optimize = self.critic_optimizer.minimize(
                loss=self.critic_loss,
                var_list=self.model.critic_trainable_vars(),
                name="critic_optimize"
            )

            self.actor_optimizer = tf.train.AdamOptimizer(
                learning_rate=actor_lr,
                epsilon=1e-8,
                name="actor_adam"
            )

            self.actor_optimize = self.actor_optimizer.minimize(
                loss=self.actor_loss,
                var_list=self.model.actor_trainable_vars(),
                name="actor_optimize"
            )

    def initialize_training(self):
        # Initialize tensorflow variables (reset model weights etc.)
        self.session.run(tf.global_variables_initializer())
        # Copy weights from model to target
        self.session.run(self.initialization_update_ops)

    def train(self, rollout):
        next_state_value = self.target_model.state_value(rollout['observations_next'])[:, 0]
        value_target = rollout['rewards'] + next_state_value * (1.0 - rollout['dones']) * self.discount_factor

        if self.ops_together:
            _, _, critic_loss_value, actor_loss_value, _ = self.session.run([
                self.critic_optimize, self.actor_optimize,
                self.critic_loss, self.actor_loss,
                self.step_update_ops
            ], feed_dict={
                self.model.observations_input: rollout['observations'],
                self.model.actions_input: rollout['actions'],
                self.value_target_input: value_target.reshape(-1, 1)
            })
        else:
            _, critic_loss_value = self.session.run([self.critic_optimize, self.critic_loss], feed_dict={
                self.model.observations_input: rollout['observations'],
                self.model.actions_input: rollout['actions'],
                self.value_target_input: value_target.reshape(-1, 1)
            })

            _, actor_loss_value = self.session.run([self.actor_optimize, self.actor_loss], feed_dict={
                self.model.observations_input: rollout['observations'],
            })

            # Update target model
            _ = self.session.run([self.step_update_ops])

        return {
            'critic_loss': critic_loss_value,
            'actor_loss': actor_loss_value
        }
