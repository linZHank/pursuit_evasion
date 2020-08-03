import sys
import os
import numpy as np
import tensorflow as tf
import scipy.signal
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


from envs.purnav_scene0 import PursuerNavigationScene0
from agents.ppo import PPOActorCritic


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:

    def __init__(self, dim_obs, dim_act, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, dim_obs[0], dim_obs[1], dim_obs[2]), dtype=np.float32)
        self.act_buf = np.zeros((size, dim_act), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        # self.ptr, self.path_start_idx = 0, 0

    def get(self, batch_size=32):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # create data dict for training actor
        actor_data = dict(
            obs = tf.convert_to_tensor(self.obs_buf, dtype=tf.float32),
            act = tf.convert_to_tensor(self.act_buf, dtype=tf.float32),
            logp = tf.convert_to_tensor(self.logp_buf, dtype=tf.float32),
            adv = tf.convert_to_tensor(self.adv_buf, dtype=tf.float32)
        )
        actor_dataset = tf.data.Dataset.from_tensor_slices(actor_data)
        batched_actor_dataset = actor_dataset.shuffle(self.max_size).batch(batch_size)
        # create data dict for training critic
        critic_data = dict(
            obs = tf.convert_to_tensor(self.obs_buf, dtype=tf.float32),
            ret = tf.convert_to_tensor(self.ret_buf, dtype=tf.float32)
        )
        critic_dataset = tf.data.Dataset.from_tensor_slices(critic_data)
        batched_critic_dataset = critic_dataset.shuffle(self.max_size).batch(batch_size)

        return batched_actor_dataset, batched_critic_dataset


if __name__=='__main__':
    # instantiate env
    env = PursuerNavigationScene0(resolution=(100,100))
    agent = PPOActorCritic(dim_obs=(100,100,3), dim_act=2, beta=0.)
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent.name)
    # parameter
    steps_per_train = 5000
    num_trains = 500
    train_epochs = 80
    max_ep_len = env.max_episode_steps
    save_freq = 50
    batch_size = 32
    # variables
    replay_buffer = PPOBuffer(dim_obs=(100,100,3), dim_act=2, size=steps_per_train)
    obs, ep_ret, ep_len = env.reset(), 0, 0
    episode_counter, step_counter = 0, 0
    success_counter = 0
    stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
    episodic_steps = []
    start_time = time.time()
    # main loop
    for t in range(num_trains):
        for st in range(steps_per_train):
            act, val, logp = agent.piofagivens(np.expand_dims(obs, axis=0)) 
            next_obs, rew, done, info = env.step(act)
            ep_ret += rew
            ep_len += 1
            stepwise_rewards.append(rew)
            step_counter += 1
            replay_buffer.store(obs, act, rew, val, logp)
            obs = next_obs # SUPER CRITICAL!!!
            # handle episode termination
            timeout = (ep_len==env.max_episode_steps)
            terminal = done or timeout
            epoch_ended = (st==steps_per_train-1)
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
                if timeout or epoch_ended:
                    _, val, _ = agent.piofagivens(np.expand_dims(obs, axis=0))
                else:
                    val = 0
                replay_buffer.finish_path(val)
                if terminal:
                    episode_counter += 1
                    episodic_returns.append(ep_ret)
                    sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                    episodic_steps.append(step_counter)
                    if info == 'Goal reached!':
                        success_counter += 1
                    print("\n====\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {} \n====\n".format(
                        step_counter, 
                        episode_counter, 
                        st+1, 
                        ep_ret, 
                        ep_len
                    ))
                obs, ep_ret, ep_len = env.reset(), 0, 0
        # Save model
        if not t%save_freq or (t==num_trains-1):
            model_path_mu = os.path.join(model_dir, 'mu', str(t))
            model_path_v = os.path.join(model_dir, 'v', str(t))
            if not os.path.exists(os.path.dirname(model_path_mu)):
                os.makedirs(os.path.dirname(model_path_mu))
            agent.actor.mu_net.save(model_path_mu)
            if not os.path.exists(os.path.dirname(model_path_v)):
                os.makedirs(os.path.dirname(model_path_v))
            agent.critic.val_net.save(model_path_v)

        # update actor-critic
        actor_dataset, critic_dataset = replay_buffer.get(batch_size=batch_size)
        loss_pi, loss_v, loss_info = agent.train(actor_dataset, critic_dataset, train_epochs)
        print("\n================================================================\nEpoch: {} \nStep: {} \nAveReturn: {} \nSucceeded: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed:{} \n================================================================\n".format(t+1, st+1, sedimentary_returns[-1], success_counter, loss_pi, loss_v, loss_info['kl'], loss_info['ent'], time.time()-start_time))
################################################################

    # Save returns 
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))
    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()


