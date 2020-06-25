""" 
A DQN type agent class for pe_env_discrete 
"""
import tensorflow as tf
import numpy as np


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

    def store(self, img, odom, act, rew, done, nxt_img, next_odom):
        self.img_buf[self.ptr] = img
        self.odom_buf[self.ptr] = odom
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.nxt_img_buf[self.ptr] = nxt_img
        self.nxt_odom_buf[self.ptr] = next_odom
        self.ptr = (self.ptr + 1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        ids = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            img = tf.convert_to_tensor(self.img_buf[ids], dtype=tf.float32),
            odom = tf.convert_to_tensor(self.odom_buf[ids], dtype=tf.float32),
            act = tf.convert_to_tensor(self.act_buf[ids], dtype=tf.int8),
            rew = tf.convert_to_tensor(self.rew_buf[ids], dtype=tf.float32),
            done = tf.convert_to_tensor(self.done_buf[ids], dtype=tf.float32),
            nxt_img = tf.convert_to_tensor(self.nxt_img_buf[ids], dtype=tf.float32),
            nxt_odom = tf.convert_to_tensor(self.nxt_odom_buf[ids], dtype=tf.float32)
        )

        return batch


def dqn(dim_img, dim_odom, dim_act):
    """
    Gives you a pe_env_discrete flavored DQN model
    """
    img_input = tf.keras.Input(shape=(dim_img[0],dim_img[1],3), name='img')
    odom_input = tf.keras.Input(shape=(dim_odom,), name='odom')
    img_feature = tf.keras.layers.Conv2D(16,3, padding='same', activation='relu')(img_input)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(img_feature)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(img_feature)
    img_feature = tf.keras.layers.Flatten()(img_feature)
    img_feature = tf.keras.layers.Dense(128, activation='relu')(img_feature)
    img_feature = tf.keras.layers.Dense(64, activation='relu')(img_feature)
    odom_feature = tf.keras.layers.Dense(16, activation='relu')(odom_input)
    cat_feature = tf.keras.layers.concatenate([img_feature, odom_feature])
    q_vals = tf.keras.layers.Dense(dim_act, activation=None, name='Q_values')(cat_feature)
    
    return tf.keras.Model(inputs=[img_input, odom_input], outputs=q_vals)

class DQNAgent:
    """
    DQN agent class. epsilon decay, epsilon greedy, train, etc..
    """
    def __init__(self, name, dim_img=(150,150,3), dim_odom=4, dim_act=5, buffer_size=int(1e5)):
        self.name = name
        # build DQN model
        self.dqn_active = dqn(dim_img=dim_img, dim_odom=dim_odom, dim_act=dim_act)
        self.dqn_active.summary()
        self.dqn_stable = tf.keras.models.clone_model(self.dqn_active)
        # build replay buffer
        self.replay_buffer = ReplayBuffer(buf_size=buffer_size, dim_img=dim_img, dim_odom=dim_odom, dim_act=dim_act)

#     def epsilon_greedy(self, state):
#         if no.random.rand() > self.epsilon:
#             action = self.dqn_active()
            


if __name__=='__main__':
    agent = DQNAgent(name='test_dqn_agent')
