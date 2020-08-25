""" 
A DQN type agent class for pe_env_discrete 
using img as input only
"""
import tensorflow as tf
import numpy as np
import logging

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
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
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
    def __init__(self, dim_obs, size):
        self.obs_buf = np.zeros([size]+list(dim_obs), dtype=np.uint8)
        self.nobs_buf = np.zeros_like(self.obs_buf)
        self.act_buf = np.zeros(shape=size, dtype=np.int)
        self.rew_buf = np.zeros(shape=size, dtype=np.float32)
        self.done_buf = np.zeros(shape=size, dtype=np.bool)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, nobs):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        ids = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs = tf.convert_to_tensor(self.obs_buf[ids]/255., dtype=tf.float32),
            nobs = tf.convert_to_tensor(self.nobs_buf[ids]/255., dtype=tf.float32),
            act = tf.convert_to_tensor(self.act_buf[ids], dtype=tf.int32),
            rew = tf.convert_to_tensor(self.rew_buf[ids], dtype=tf.float32),
            done = tf.convert_to_tensor(self.done_buf[ids], dtype=tf.float32),
        )

        return batch

class Critic(tf.keras.Model):
    def __init__(self, dim_obs, num_act, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        img_inputs = tf.keras.Input(shape=dim_obs, name='img_inputs')
        # image features
        img_feature = tf.keras.layers.Conv2D(32,3,activation=activation)(img_inputs)
        img_feature = tf.keras.layers.Conv2D(32,3,activation=activation)(img_feature)
        img_feature = tf.keras.layers.Conv2D(32,3,activation=activation)(img_feature)
        img_feature = tf.keras.layers.Flatten()(img_feature)
        img_feature = tf.keras.layers.Dense(128, activation=activation)(img_feature)
        outputs = tf.keras.layers.Dense(num_act, activation=None, name='Q_values')(img_feature)
        self.q_net = tf.keras.Model(inputs=img_inputs, outputs=outputs)
        
    def call(self, obs):
        return self.q_net(obs)
        # return tf.squeeze(qval, axis=-1)

class DeepQNet(tf.keras.Model):
    def __init__(self, dim_obs, num_act, activation='relu', gamma = 0.99, alpha=0.,
                 lr=3e-4, polyak=0.995, **kwargs):
        super(DeepQNet, self).__init__(name='dqn', **kwargs)
        # params
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.gamma = gamma # discount rate
        self.polyak = polyak
        self.init_eps = 1.
        self.final_eps = .1
        # model
        self.q = Critic(dim_obs, num_act, activation) 
        self.targ_q = Critic(dim_obs, num_act, activation)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        # variable
        self.epsilon = self.init_eps

    def linear_epsilon_decay(self, episode, decay_period, warmup_episodes):
        episodes_left = decay_period + warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    def act(self, obs):
        if obs.dtype==np.uint8:
            obs = obs.astype(np.float32)/255.
        if np.random.rand() > self.epsilon:
            a = tf.argmax(self.q(obs), axis=-1)
        else:
            a = tf.random.uniform(shape=[1,1], maxval=self.num_act, dtype=tf.dtypes.int32)
        return a

    def train_one_batch(self, data):
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_weights)
            pred_qval = tf.math.reduce_sum(self.q(data['obs']) * tf.one_hot(data['act'], self.num_act), axis=-1)
            targ_qval = data['rew'] + self.gamma*(1-data['done'])*tf.math.reduce_sum(self.targ_q(data['nobs'])*tf.one_hot(tf.math.argmax(self.q(data['nobs']),axis=1), self.num_act),axis=1) # double DQN trick
            loss_q = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval)
        grads = tape.gradient(loss_q, self.q.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_weights))
        # Polyak average update target Q-nets
        q_weights_update = []
        for w_q, w_targ_q in zip(self.q.get_weights(), self.targ_q.get_weights()):
            w_q_upd = self.polyak*w_targ_q
            w_q_upd = w_q_upd + (1 - self.polyak)*w_q
            q_weights_update.append(w_q_upd)
        self.targ_q.set_weights(q_weights_update)

        return loss_q


if __name__=='__main__':
    agent = DeepQNet(dim_obs=[80,80,3], num_act=5)
    buf = ReplayBuffer(size=128, dim_obs=[80,80,3])
    obs = np.random.randint(0, 255, (128,80,80,3), dtype=np.uint8)
    nobs = np.random.randint(0, 255, (128,80,80,3), dtype=np.uint8)
    act = np.random.randint(5, size=128)
    rew = np.random.randn(128).astype(np.float32)
    done = 1.*np.random.choice([False, True], size=128)
    for i in range(128):
        buf.store(obs[i], act[i], rew[i], done[i], nobs[i])
    batch = buf.sample_batch()
    loss_q = agent.train_one_batch(batch)

