""" 
A DQN type agent class for pe_env_discrete 
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
    def __init__(self, buf_size, dim_img, dim_odom, dim_act):
        self.img_buf = np.zeros(shape=(buf_size, dim_img[0], dim_img[1], dim_img[2]), dtype=np.float32)
        self.odom_buf = np.zeros(shape=(buf_size, dim_odom), dtype=np.float32)
        self.nxt_img_buf = np.zeros(shape=(buf_size, dim_img[0], dim_img[1], dim_img[2]), dtype=np.float32)
        self.nxt_odom_buf = np.zeros(shape=(buf_size, dim_odom), dtype=np.float32)
        self.act_buf = np.zeros(shape=buf_size, dtype=np.int)
        self.rew_buf = np.zeros(shape=buf_size, dtype=np.float32)
        self.done_buf = np.zeros(shape=buf_size, dtype=np.bool)
        self.ptr, self.size, self.max_size = 0, 0, buf_size

    def store(self, img, odom, act, rew, done, nxt_img, nxt_odom):
        self.img_buf[self.ptr] = img
        self.odom_buf[self.ptr] = odom
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.nxt_img_buf[self.ptr] = nxt_img
        self.nxt_odom_buf[self.ptr] = nxt_odom
        self.ptr = (self.ptr + 1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        ids = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            img = tf.convert_to_tensor(self.img_buf[ids], dtype=tf.float32),
            odom = tf.convert_to_tensor(self.odom_buf[ids], dtype=tf.float32),
            act = tf.convert_to_tensor(self.act_buf[ids], dtype=tf.int32),
            rew = tf.convert_to_tensor(self.rew_buf[ids], dtype=tf.float32),
            done = tf.convert_to_tensor(self.done_buf[ids], dtype=tf.float32),
            nxt_img = tf.convert_to_tensor(self.nxt_img_buf[ids], dtype=tf.float32),
            nxt_odom = tf.convert_to_tensor(self.nxt_odom_buf[ids], dtype=tf.float32)
        )

        return batch

def dqn(dim_img, dim_odom, dim_act, activation='relu'):
    """
    Gives you a pe_env_discrete flavored DQN model
    """
    img_input = tf.keras.Input(shape=(dim_img[0],dim_img[1],3), name='img')
    odom_input = tf.keras.Input(shape=(dim_odom,), name='odom')
    # image features
    img_feature = tf.keras.layers.Conv2D(16,3, padding='same', activation=activation)(img_input)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Conv2D(32, 3, padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Conv2D(64, 3, padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Flatten()(img_feature)
    img_feature = tf.keras.layers.Dense(512, activation=activation)(img_feature)
    # odom features
    odom_feature = tf.keras.layers.Dense(16, activation=activation)(odom_input)
    odom_feature = tf.keras.layers.Dense(16, activation=activation)(odom_feature)
    # concatenate features
    cat_feature = tf.keras.layers.concatenate([img_feature, odom_feature])
    q_vals = tf.keras.layers.Dense(dim_act, activation=None, name='Q_values')(cat_feature)
    
    return tf.keras.Model(inputs=[img_input, odom_input], outputs=q_vals) 

class DQNAgent:
    """
    DQN agent class. epsilon decay, epsilon greedy, train, etc..
    """
    def __init__(self, name='dqn_agent', dim_img=(150,150,3), dim_odom=4, dim_act=5, buffer_size=int(1e6),
                 decay_period=500,
                 warmup_episodes=100, init_epsilon=1., final_epsilon=.1, learning_rate=3e-4,
                 loss_fn=tf.keras.losses.MeanSquaredError(), batch_size=64, discount_rate=0.99, sync_step=4096):
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
        self.dqn_active = dqn(dim_img=dim_img, dim_odom=dim_odom, dim_act=dim_act)
        self.dqn_active.summary()
        self.dqn_stable = tf.keras.models.clone_model(self.dqn_active)
        # build replay buffer
        self.replay_buffer = ReplayBuffer(buf_size=buffer_size, dim_img=dim_img, dim_odom=dim_odom, dim_act=dim_act)

    def epsilon_greedy(self, img, odom):
        if np.random.rand() > self.epsilon:
            vals = self.dqn_active([np.expand_dims(img, axis=0), np.expand_dims(odom, axis=0)])
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
            vals = self.dqn_active([minibatch['img'], minibatch['odom']])
            oh_acts = tf.one_hot(minibatch['act'], depth=self.dim_act)
            pred_qvals = tf.math.reduce_sum(tf.math.multiply(vals, oh_acts), axis=-1)
            # compute target Q
            nxt_vals = self.dqn_stable([minibatch['nxt_img'], minibatch['nxt_odom']])
            nxt_acts = tf.math.argmax(self.dqn_active([minibatch['nxt_img'], minibatch['nxt_odom']]), axis=-1)
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


if __name__=='__main__':
    agent = DQNAgent(name='test_dqn_agent')
    test_img = np.random.rand(4,150,150,3)
    test_odom = np.random.randn(4,4)
    qvals = agent.dqn_active([test_img, test_odom])
    print("qvals: {}".format(qvals))

