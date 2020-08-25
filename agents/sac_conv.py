import sys
import os
from copy import deepcopy
import numpy as np
import random
import time
from datetime import datetime
import logging

import tensorflow as tf
print(tf.__version__)
import tensorflow_probability as tfp
tfd = tfp.distributions
################################################################
"""
Unnecessary initial settings
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
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
################################################################


def convnet(dim_inputs, dim_outputs, activation, output_activation=None):
    # inputs
    img_inputs = tf.keras.Input(shape=dim_inputs, name='img_inputs')
    # image features
    img_feature = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation=activation)(img_inputs)
    img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.Flatten()(img_feature)
    img_feature = tf.keras.layers.Dense(128, activation=activation)(img_feature)
    img_feature = tf.keras.layers.Dense(128, activation=activation)(img_feature)
    # outputs
    outputs = tf.keras.layers.Dense(dim_outputs, activation=output_activation)(img_feature)

    return tf.keras.Model(inputs=img_inputs, outputs=outputs)

class Actor(tf.keras.Model):
    def __init__(self, dim_obs, dim_act, activation, act_limit, **kwargs):
        super(Actor, self).__init__(name='actor', **kwargs)
        # model construction
        img_inputs = tf.keras.Input(shape=dim_obs, name='img_inputs')
        img_feature = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation=activation)(img_inputs)
        img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
        img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
        img_feature = tf.keras.layers.Flatten()(img_feature)
        img_feature = tf.keras.layers.Dense(256, activation=activation)(img_feature)
        img_feature = tf.keras.layers.Dense(128, activation=activation)(img_feature)
        mu_outputs = tf.keras.layers.Dense(dim_act[0])(img_feature) 
        log_std_outputs = tf.keras.layers.Dense(dim_act[0])(img_feature) 
        # instantiate model
        self.policy_net = tf.keras.Model(inputs=img_inputs, outputs=[mu_outputs, log_std_outputs])
        self.act_limit = act_limit

    def call(self, obs, deterministic=False, with_logprob=True):
        mu, log_std = self.policy_net(obs)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.math.exp(log_std)
        pi_distribution = tfd.Normal(mu, std)
        if deterministic:
            action = mu # only use for evaluation
        else: # reparameterization trick
            eps = tf.random.normal(shape=mu.shape)
            action = mu + eps*std 
        if with_logprob:
            # arXiv 1801.01290, appendix C
            logp_pi = tf.math.reduce_sum(pi_distribution.log_prob(action), axis=-1)
            logp_pi -= tf.math.reduce_sum(2*(np.log(2) - action - tf.math.softplus(-2*action)), axis=-1)
        else:
            logp_pi = None
        action = tf.math.tanh(action)
        action = self.act_limit*action

        return action, logp_pi
        
class Critic(tf.keras.Model):
    def __init__(self, dim_obs, dim_act, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        img_inputs = tf.keras.Input(shape=dim_obs, name='img_inputs')
        act_inputs = tf.keras.Input(shape=dim_act, name='act_inputs')
        img_feature = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation=activation)(img_inputs)
        img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
        img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
        img_feature = tf.keras.layers.Flatten()(img_feature)
        img_feature = tf.keras.layers.Dense(256, activation=activation)(img_feature)
        img_feature = tf.keras.layers.Dense(128, activation=activation)(img_feature)
        concat_feature = tf.keras.layers.concatenate([img_feature, act_inputs])
        outputs = tf.keras.layers.Dense(1)(concat_feature)
        self.q_net = tf.keras.Model(inputs=[img_inputs,act_inputs], outputs=outputs)
        
    def call(self, obs, act):
        qval = self.q_net([obs, act])
        return tf.squeeze(qval, axis=-1)

class SoftActorCritic(tf.keras.Model):

    def __init__(self, dim_obs, dim_act, act_limit, activation='relu', gamma = 0.99, auto_ent=True,
                 alpha=0.2, critic_lr=1e-4, actor_lr=1e-4, alpha_lr=1e-4, polyak=0.995, **kwargs):
        super(SoftActorCritic, self).__init__(name='sac_conv', **kwargs)
        # params
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.auto_ent = auto_ent
        self.target_ent = -np.prod(dim_act) # heuristic
        self.alpha = alpha # entropy temperature
        self.gamma = gamma # discount rate
        self.polyak = polyak
        # models
        self.pi = Actor(dim_obs, dim_act, activation, act_limit)
        self.q0 = Critic(dim_obs, dim_act, activation) 
        self.q1 = Critic(dim_obs, dim_act, activation) 
        self.targ_q0 = Critic(dim_obs, dim_act, activation)
        self.targ_q1 = Critic(dim_obs, dim_act, activation)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=actor_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=alpha_lr)
        # variables
        self._log_alpha = tf.Variable(0.0)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)

    def act(self, obs, deterministic=False):
        a, _ = self.pi(obs, deterministic, False)
        return a.numpy()

    def train_one_batch(self, data):
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.q0.trainable_weights+self.q1.trainable_weights)
            pred_qval0 = self.q0(data['obs'], data['act'])
            pred_qval1 = self.q1(data['obs'], data['act'])
            nact, nlogp = self.pi(data['nobs'])
            nqval0 = self.targ_q0(data['nobs'], nact) # compute qval for next step
            nqval1 = self.targ_q1(data['nobs'], nact)
            pessi_nqval = tf.math.minimum(nqval0, nqval1) # pessimistic value
            targ_qval = data['rew'] + self.gamma*(1 - data['done'])*(pessi_nqval - self.alpha*nlogp)
            loss_q0 = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval0)
            loss_q1 = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval1)
            loss_q = loss_q0 + loss_q1
        grads_critic = tape.gradient(loss_q, self.q0.trainable_weights+self.q1.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grads_critic, self.q0.trainable_weights+self.q1.trainable_weights))
        # update actor
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_weights)
            act, logp = self.pi(data['obs'])
            qval0 = self.q0(data['obs'], act)
            qval1 = self.q1(data['obs'], act)
            pessi_qval = tf.math.minimum(qval0, qval1)
            loss_pi = tf.math.reduce_mean(self.alpha*logp - pessi_qval)
        grads_actor = tape.gradient(loss_pi, self.pi.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(grads_actor, self.pi.trainable_weights))
        # Polyak average update target Q-nets
        q0_weights_update = []
        for w_q0, w_targ_q0 in zip(self.q0.get_weights(), self.targ_q0.get_weights()):
            w_q0_upd = self.polyak*w_targ_q0
            w_q0_upd = w_q0_upd + (1 - self.polyak)*w_q0
            q0_weights_update.append(w_q0_upd)
        self.targ_q0.set_weights(q0_weights_update)
        q1_weights_update = []
        for w_q1, w_targ_q1 in zip(self.q1.get_weights(), self.targ_q1.get_weights()):
            w_q1_upd = self.polyak*w_targ_q1
            w_q1_upd = w_q1_upd + (1 - self.polyak)*w_q1
            q1_weights_update.append(w_q1_upd)
        self.targ_q1.set_weights(q1_weights_update)
        # update alpha
        if self.auto_ent:
            with tf.GradientTape() as tape:
                tape.watch([self._log_alpha])
                _, logp = self.pi(data['obs'])
                loss_alpha = -tf.math.reduce_mean(self._alpha*logp+self.target_ent)
            grads_alpha = tape.gradient(loss_alpha, [self._log_alpha])
            self.alpha_optimizer.apply_gradients(zip(grads_alpha, [self._log_alpha]))
            self.alpha = self._alpha.numpy()

        return loss_q, loss_pi
    
# Test agent
if __name__=='__main__':
    agent = SoftActorCritic(dim_obs=(100,100,1), dim_act=(2,), act_limit=2)
    # test dataset
    img = np.random.randint(0, 255, (8,100,100,1), dtype=np.uint8)
    nimg = np.random.randint(0, 255, (8,100,100,1), dtype=np.uint8)
    act = np.random.randn(8,2).astype(np.float32)
    rew = np.random.randn(8).astype(np.float32)
    done = 1.*np.random.choice([False, True], size=8)
    data = dict(
        obs = tf.convert_to_tensor(img/255., dtype=tf.float32),
        nobs = tf.convert_to_tensor(nimg/255., dtype=tf.float32),
        act = tf.convert_to_tensor(act, dtype=tf.float32),
        rew = tf.convert_to_tensor(rew, dtype=tf.float32),
        done = tf.convert_to_tensor(done, dtype=tf.float32)
    )
    # loss_q, loss_pi = agent.train_one_batch(data)
    # print("loss_q: {}, loss_pi: {}".format(loss_q, loss_pi))
