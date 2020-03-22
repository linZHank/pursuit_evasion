"""
DQN class, multiple instances training enabled
"""


from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import random
import pickle
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class Memory:
    """
    This class defines replay buffer
    """
    def __init__(self, memory_cap):
        self.memory_cap = memory_cap
        self.memory = []
    def store(self, experience):
        # pop a random experience if memory full
        if len(self.memory) >= self.memory_cap:
            self.memory.pop(random.randint(0, len(self.memory)-1))
        self.memory.append(experience)
        print("experience: {} stored to memory".format(experience))

    def sample_batch(self, batch_size):
        # Select batch
        if len(self.memory) < batch_size:
            batch = random.sample(self.memory, len(self.memory))
        else:
            batch = random.sample(self.memory, batch_size)
        print("A batch of memories are sampled with size: {}".format(batch_size))

        return list(zip(*batch))


class DQNAgent:
    def __init__(self):
        # fixed
        self.name = 'pursuer'
        self.dim_state = 4
        self.actions = np.array([[-1,-1],[1,-1],[-1,1],[1,1]]) # [d_x,d_y]
        # hyper-parameters
        self.memory_cap = 100000
        self.layer_sizes = [64,32]
        self.update_step = 8192
        self.learning_rate = 0.001
        self.batch_size = 4096
        self.gamma = 0.999
        self.init_eps = 1.
        self.final_eps = 0.1
        self.warmup_episodes = 0
        # variables
        self.epsilon = 1
        self.epoch_counter = 0
        # Q(s,a;theta)
        assert len(self.layer_sizes) >= 1
        inputs = tf.keras.Input(shape=(self.dim_state,), name='state')
        x = layers.Dense(self.layer_sizes[0], activation='relu')(inputs)
        for i in range(1,len(self.layer_sizes)):
            x = layers.Dense(self.layer_sizes[i], activation='relu')(x)
        outputs = layers.Dense(self.actions.shape[0])(x)
        self.qnet_active = Model(inputs=inputs, outputs=outputs, name='qnet_model')
        # clone active Q-net to create stable Q-net
        self.qnet_stable = tf.keras.models.clone_model(self.qnet_active)
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        # loss function
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        # metrics
        self.mse_metric = keras.metrics.MeanSquaredError()
        # init replay memory
        self.replay_memory = Memory(memory_cap=self.memory_cap)

    def epsilon_greedy(self, state):
        """
        If a random number(0,1) beats epsilon, return index of largest Q-value.
        Else, return a random index
        """
        if np.random.rand() > self.epsilon:
            index = np.argmax(self.qnet_active(state.reshape(1,-1)))
        else:
            index = np.random.randint(self.actions.shape[0])
            print("{} Take a random action!".format(self.name))
        action = self.actions[index]

        return index, action

    def linear_epsilon_decay(self, episode, decay_period):
        """
        Returns the current epsilon for the agent's epsilon-greedy policy. This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et al., 2015). The schedule is as follows:
            Begin at 1. until warmup_steps steps have been taken; then Linearly decay epsilon from 1. to final_eps in decay_period steps; and then Use epsilon from there on.
        Args:
            decay_period: int
            episode: int
        Returns:
        """
        episodes_left = decay_period + self.warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    def exponential_epsilon_decay(self, episode, decay_rate):
        """
        Returns the current epsilon for the agent's epsilon-greedy policy:
            Begin at 1. until warmup_steps steps have been taken; then exponentially decay epsilon from 1. to final_eps; and then Use epsilon from there on.
        Args:
            episode: int, the number of training steps completed so far.
            decay_rate: exponential rate of epsilon decay
        Returns:
        """
        if episode >= self.warmup_episodes:
            self.epsilon *= decay_rate
        if self.epsilon <= self.final_eps:
            self.epsilon = self.final_eps

    def train(self):
        # sample a minibatch from replay buffer
        minibatch = self.replay_memory.sample_batch(self.batch_size)
        (batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states) = [np.array(minibatch[i]) for i in range(len(minibatch))]
        # open a GradientTape to record the operations run during the forward pass
        with tf.GradientTape() as tape:
            # run forward pass
            # pred_q = tf.math.reduce_sum(tf.cast(self.qnet_active(batch_states), tf.float32) * tf.one_hot(batch_actions, self.actions.shape[0]), axis=-1)
            pred_q = tf.math.reduce_sum(self.qnet_active(batch_states) * tf.one_hot(batch_actions, self.actions.shape[0]), axis=-1)
            target_q = batch_rewards + (1. - batch_done_flags) * self.gamma * tf.math.reduce_sum(self.qnet_stable(batch_next_states)*tf.one_hot(tf.math.argmax(self.qnet_active(batch_next_states),axis=1), self.actions.shape[0]),axis=1)
            # compute loss value
            loss_value = self.loss_fn(y_true=target_q, y_pred=pred_q)
        # use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.qnet_active.trainable_weights)
        # run one step of gradient descent
        self.optimizer.apply_gradients(zip(grads, self.qnet_active.trainable_weights))
        # update metrics
        self.mse_metric(target_q, pred_q)
        # display metrics
        train_mse = self.mse_metric.result()
        print("{} mse: {}".format(self.name, train_mse))
        # reset training metrics
        self.mse_metric.reset_states()
        # epoch_counter ++
        self.epoch_counter += 1

    def save_model(self, model_dir):
        self.qnet_active.summary()
        self.qnet_stable.summary()
        # create model saving directory if not exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # save model
        self.qnet_active.save(os.path.join(model_dir, 'active_model-'+str(self.epoch_counter)+'.h5'))
        self.qnet_stable.save(os.path.join(model_dir, 'stable_model-'+str(self.epoch_counter)+'.h5'))
        print("Q_net models saved at {}".format(model_dir))

    def load_model(self, model_path):
        self.qnet_active = tf.keras.models.load_model(model_path)
        self.qnet_stable = tf.keras.models.clone_model(self.qnet_active)
        print("Q-Net models loaded")
        self.qnet_active.summary()

    def save_memory(self, memory_dir):
        # save transition buffer memory
        data_utils.save_pkl(content=self.replay_memory, fdir=memory_dir, fname='memory.pkl')
        print("transitions memory saved at {}".format(memory_dir))

    def load_memory(self, memory_path):
        with open(memory_path, 'rb') as f:
            self.replay_memory = pickle.load(f)
        print("Replay Buffer Loaded")
