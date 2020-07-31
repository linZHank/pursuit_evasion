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

from envs.pur_nav import PursuerNavigation
from agents.sac import SoftActorCritic


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, dim_obs, dim_act, size):
        self.obs_buf = np.zeros((size, dim_obs[0], dim_obs[1], dim_obs[2]), dtype=np.float32)
        self.nobs_buf = np.zeros_like(self.obs_buf)
        self.act_buf = np.zeros((size, dim_act), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, nobs):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=tf.convert_to_tensor(self.obs_buf[idxs]),
                     nobs=tf.convert_to_tensor(self.nobs_buf[idxs]),
                     act=tf.convert_to_tensor(self.act_buf[idxs]),
                     rew=tf.convert_to_tensor(self.rew_buf[idxs]),
                     done=tf.convert_to_tensor(self.done_buf[idxs]))
        return batch

if __name__=='__main__':
    env = PursuerNavigation(resolution=(80,80))
    sac = SoftActorCritic(dim_obs=(80,80,3), dim_act=2, auto_ent=True)
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, sac.name)
    # params
    max_episode_steps = env.max_episode_steps
    batch_size = 32
    update_freq = 50
    update_after = 1000
    warmup_steps = 5000
    replay_buffer = ReplayBuffer(dim_obs=(80,80,3), dim_act=2, size=int(1e5)) 
    total_steps = int(4e5)
    episodic_returns = []
    sedimentary_returns = []
    episodic_steps = []
    save_freq = 100
    episode_counter = 0
    obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
    start_time = time.time()
    for t in range(total_steps):
        # env.render()
        if t < warmup_steps:
            act = np.random.uniform(env.action_space_low, env.action_space_high, size=2)
        else:
            act = np.squeeze(sac.act(np.expand_dims(obs, axis=0)))
        nobs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1
        done = False if ep_len == max_episode_steps else done
        replay_buffer.store(obs, act, rew, done, nobs)
        obs = nobs
        if done or (ep_len==max_episode_steps):
            episode_counter += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/episode_counter)
            episodic_steps.append(t+1)
            print("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpisodeReturn: {} \nSedimentaryReturn: {} \nTimeElapsed: {} \n====\n".format(
                episode_counter, ep_len, t+1, ep_ret, sedimentary_returns[-1],
                time.time()-start_time
            ))
            # save model
            if not episode_counter%save_freq:
                model_path = os.path.join(model_dir, str(episode_counter))
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                sac.pi.policy_net.save(model_path)
            # reset env
            obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
        if not t%update_freq and t>=update_after:
            for _ in range(update_freq*4):
                minibatch = replay_buffer.sample_batch(batch_size=batch_size)
                loss_q, loss_pi = sac.train_one_batch(data=minibatch)
                logging.debug("\nloss_q: {} \nloss_pi: {} \nalpha: {}".format(loss_q, loss_pi, sac.alpha))

    # Save returns 
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))
    # Save final model
    model_path = os.path.join(model_dir, str(episode_counter))
    sac.pi.policy_net.save(model_path)
    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()

# Test
input("Press ENTER to test lander...")
for ep in range(10):
    o, d, ep_ret = env.reset(), False, 0
    for st in range(max_episode_steps):
        env.render()
        a = np.squeeze(sac.act(np.expand_dims(obs, axis=0), deterministic=True))
        o2,r,d,_ = env.step(a)
        ep_ret += r
        o = o2
        if d:
            print("EpReturn: {}".format(ep_ret))
            break 
