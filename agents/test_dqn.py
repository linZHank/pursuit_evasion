"""
Test DQN agent in OpenAI gym env
"""
import sys
import os
import tensorflow as tf
import numpy as np
import gym
import logging
from datetime import datetime
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

################################################################
"""
Can safely ignore this block
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
################################################################
class ReplayBuffer:
    """
    An off-policy replay buffer for DQN agent
    """
    def __init__(self, buf_size, dim_obs, dim_act):
        self.obs_buf = np.zeros(shape=(buf_size, dim_obs), dtype=np.float32)
        self.nxt_obs_buf = np.zeros(shape=(buf_size, dim_obs), dtype=np.float32)
        self.act_buf = np.zeros(shape=buf_size, dtype=np.int)
        self.rew_buf = np.zeros(shape=buf_size, dtype=np.float32)
        self.done_buf = np.zeros(shape=buf_size, dtype=np.bool)
        self.ptr, self.size, self.max_size = 0, 0, buf_size

    def store(self, obs, act, rew, done, nxt_obs):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.nxt_obs_buf[self.ptr] = nxt_obs
        self.ptr = (self.ptr + 1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        ids = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs = tf.convert_to_tensor(self.obs_buf[ids], dtype=tf.float32),
            act = tf.convert_to_tensor(self.act_buf[ids], dtype=tf.int32),
            rew = tf.convert_to_tensor(self.rew_buf[ids], dtype=tf.float32),
            done = tf.convert_to_tensor(self.done_buf[ids], dtype=tf.float32),
            nxt_obs = tf.convert_to_tensor(self.nxt_obs_buf[ids], dtype=tf.float32)
        )

        return batch

class DQNAgent:
    """
    DQN agent class. epsilon decay, epsilon greedy, train, etc..
    """
    def __init__(self, name='dqn_agent', dim_obs=4, dim_act=2, buffer_size=int(1e6),
                 decay_period=100,
                 warmup_episodes=100, init_epsilon=1., final_epsilon=.1, learning_rate=1e-3,
                 loss_fn=tf.keras.losses.MeanSquaredError(), batch_size=64, discount_rate=0.99, sync_step=1024):
        # hyper parameters
        self.name = name
        self.dim_act = dim_act
        self.decay_period = decay_period
        self.warmup_episodes = warmup_episodes
        self.init_epsilon = init_epsilon
        self.learning_rate = learning_rate
        self.final_epsilon = final_epsilon
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.gamma = discount_rate
        self.sync_step = sync_step
        # variables
        self.epsilon = 1.
        self.fit_cntr = 0
        # build DQN model
        ## test qnet, for testing in openai gym
        self.dqn_active = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(dim_obs,)), # CartPole
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(dim_act)
            ]
        )
        self.dqn_active.summary()
        self.dqn_stable = tf.keras.models.clone_model(self.dqn_active)
        # build replay buffer
        self.replay_buffer = ReplayBuffer(buf_size=buffer_size, dim_obs=dim_obs, dim_act=dim_act)

    def epsilon_greedy(self, obs):
        if np.random.rand() > self.epsilon:
            vals = self.dqn_active(np.expand_dims(obs, axis=0))
            action = np.argmax(vals)
        else:
            action = np.random.randint(self.dim_act)
            logging.warning("{} performs a random action: {}".format(self.name, action))

        return action
            
    def linear_epsilon_decay(self, curr_ep):
        """
        Begin at 1. until warmup_steps steps have been taken; then Linearly decay epsilon from 1. to final_eps in decay_period steps; and then Use epsilon from there on.
        Args:
            curr_ep: current episode index
        Returns:
            current epsilon for the agent's epsilon-greedy policy
        """
        episodes_left = self.decay_period + self.warmup_episodes - curr_ep
        bonus = (self.init_epsilon - self.final_epsilon) * episodes_left / self.decay_period
        bonus = np.clip(bonus, 0., self.init_epsilon-self.final_epsilon)
        self.epsilon = self.final_epsilon + bonus

    # @tf.function
    def train_one_step(self):
        minibatch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        with tf.GradientTape() as tape:
            # compute current Q
            vals = self.dqn_active(minibatch['obs'])
            oh_acts = tf.one_hot(minibatch['act'], depth=self.dim_act)
            pred_qvals = tf.math.reduce_sum(tf.math.multiply(vals, oh_acts), axis=-1)
            # compute target Q
            nxt_vals = self.dqn_stable(minibatch['nxt_obs'])
            nxt_acts = tf.math.argmax(self.dqn_active(minibatch['nxt_obs']), axis=-1)
            oh_nxt_acts = tf.one_hot(nxt_acts, depth=self.dim_act)
            nxt_qvals = tf.math.reduce_sum(tf.math.multiply(nxt_vals, oh_nxt_acts), axis=-1)
            targ_qvals = minibatch['rew'] + (1. - minibatch['done'])*self.gamma*nxt_qvals
            # compute loss
            loss_q = self.loss_fn(y_true=targ_qvals, y_pred=pred_qvals)
            logging.info("loss_Q: {}".format(loss_q))
        # gradient decent
        grads = tape.gradient(loss_q, self.dqn_active.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.dqn_active.trainable_weights))
        self.fit_cntr += 1
        # update dqn_stable if C steps of q_val fitted
        if not self.fit_cntr%self.sync_step:
            self.dqn_stable.set_weights(self.dqn_active.get_weights())



env = gym.make('CartPole-v1')
agent = DQNAgent()
# parameters
num_episodes = 300
num_steps = env.spec.max_episode_steps
step_counter = 0
train_freq = 10
model_dir = './test_training_models/dqn/'+datetime.now().strftime("%Y-%m-%d-%H-%M")
# vars
ep_rets = []
ave_rets = []
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
ep_rets, ave_rets = [], []
if __name__ == '__main__':
    for ep in range(num_episodes):
        obs, done, ep_rew = env.reset(), False, 0
        agent.linear_epsilon_decay(curr_ep=ep)
        for st in range(num_steps):
            # env.render()
            act = agent.epsilon_greedy(obs)
            next_obs, rew, done, info = env.step(act)
            agent.replay_buffer.store(obs, act, rew, done, next_obs)
            if ep >= agent.warmup_episodes:
                if not step_counter%train_freq:
                    for _ in range(train_freq):
                        agent.train_one_step()
            # update qnet_stable every update_steps
            ep_rew += rew
            step_counter += 1
            logging.debug("\n-\nepisode: {}, step: {}, \nobs: {} \naction: {} \nreward: {}".format(ep+1, st+1, obs, act,rew))
            obs = next_obs.copy()
            if done or st==num_steps-1:
                ep_rets.append(ep_rew)
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                logging.info("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rew, ave_rets[-1]))
                break
        # ep_rets.append(ep_rew)
        # ave_rets.append(sum(ep_rets)/len(ep_rets))
        # logging.info("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rew, ave_rets[-1]))
# Save final ckpt
# save_path = ckpt_manager.save()
# save model
# qnet_active.save(os.path.join(model_dir, str(step_counter)+'.h5'))

# Plot returns and loss
fig, axes = plt.subplots(2, figsize=(12, 8))
fig.suptitle('Metrics')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Averaged Return")
axes[0].plot(ave_rets)
# axes[1].set_xlabel("Steps")
# axes[1].set_ylabel("Loss")
# axes[1].plot(train_loss)
plt.show()

