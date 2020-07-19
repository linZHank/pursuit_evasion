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


from envs.pe_1v1_continuous import PursuitEvasionOneVsOneContinuous
from agents.ppo import PPOAgent


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
    def __init__(self, size, gamma=.99, lam=.95):
        self.img_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.adv_buf = []
        self.val_buf = []
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.episode_start_idx, self.max_size = 0, 0, size

    def store(self, img, act, logp, rew, val):
        assert self.ptr <= self.max_size
        self.img_buf.append(img)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.ptr += 1

    def finish_episode(self, last_val=0):
        ep_slice = slice(self.episode_start_idx, self.ptr)
        rews = np.array(self.rew_buf[ep_slice])
        vals = np.append(np.array(self.val_buf[ep_slice]), last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf += list(discount_cumsum(deltas, self.gamma*self.lam))
        # next line implement reward-to-go
        self.ret_buf += list(discount_cumsum(np.append(rews, last_val), self.gamma)[:-1])
        self.episode_start_idx = self.ptr

    def get(self):
        """
        Get a data dicts from replay buffer
        """
        # convert list to array
        img_buf = np.array(self.img_buf)
        act_buf = np.array(self.act_buf) 
        logp_buf = np.array(self.logp_buf)
        rew_buf = np.array(self.rew_buf) 
        ret_buf = np.array(self.ret_buf) 
        adv_buf = np.array(self.adv_buf) 
        # next three lines implement advantage normalization
        adv_mean = np.mean(adv_buf)
        adv_std = np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        # create data dict for training actor
        actor_data = dict(
            img = tf.convert_to_tensor(img_buf, dtype=tf.float32),
            act = tf.convert_to_tensor(act_buf, dtype=tf.float32),
            logp = tf.convert_to_tensor(logp_buf, dtype=tf.float32),
            adv = tf.convert_to_tensor(adv_buf, dtype=tf.float32)
        )
        actor_dataset = tf.data.Dataset.from_tensor_slices(actor_data)
        # create data dict for training critic
        critic_data = dict(
            img = tf.convert_to_tensor(img_buf, dtype=tf.float32),
            ret = tf.convert_to_tensor(ret_buf, dtype=tf.float32)
        )
        critic_dataset = tf.data.Dataset.from_tensor_slices(critic_data)

        return actor_dataset, critic_dataset


if __name__=='__main__':
    # instantiate env
    env = PursuitEvasionOneVsOneContinuous(resolution=(80,80))
    agent = PPOAgent(name='ppo_train', dim_img=(80,80,4), lr_actor=3e-4, lr_critic=1e-3, batch_size=128, target_kl=0.2)
    model_dir_actor = os.path.join(sys.path[0], 'saved_models', env.name, agent.name, 'models', 'actor/')
    model_dir_critic = os.path.join(sys.path[0], 'saved_models', env.name, agent.name, 'models', 'critic/')
    # parameter
    num_episodes = 100000
    num_steps = env.max_episode_steps
    buffer_size = int(3e5)
    update_every = 300
    # variables
    step_counter = 0
    episode_counter = 0
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    buf = PPOBuffer(size=buffer_size)
    # instantiate agent
    start_time = time.time()
    for ep in range(num_episodes):
        obs, ep_rew = env.reset(), 0
        for st in range(num_steps):
            # render for debug purpose
            # env.render(pause=1./env.rate)
            # cv2.imshow('map', obs[:,:,[2,1,0]])
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
            img = obs.copy()
            img = np.concatenate((img, np.expand_dims(img[:,:,-1],axis=-1)), axis=-1)
            actions = np.zeros((2,2))
            act, val, logp = agent.pi_of_a_given_s(np.expand_dims(img, axis=0))
            actions[1] = act
            n_obs, rew, done, info = env.step(actions)
            logging.debug("\n-\nepisode: {}, step: {} \naction: {} \nreward: {} \ndone: {} \ninfo: {} \n-\n".format(
                ep+1, 
                st+1, 
                act,
                rew[1], 
                done[1],
                info
            ))
            ep_rew += rew[1]
            step_counter += 1
            # store transition
            buf.store(img, act, logp, rew[1], val)
            obs = n_obs.copy() # EXTREMELY IMPORTANT!!! 
            # finish episode 
            if info:
                episode_counter += 1
                if not any(done):
                    img = obs.copy()
                    img = np.concatenate((img, np.expand_dims(img[:,:,-1],axis=-1)), axis=-1)
                    _, val, _ = agent.pi_of_a_given_s(np.expand_dims(img, axis=0))
                else:
                    val = 0
                    if done[0]:
                        success_counter += 1
                buf.finish_episode(last_val=val)
                episodic_returns.append(ep_rew)
                sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                logging.info(
                    "\n================\nEpisode: {} \nEpLength: {} \nTotalReward: {} \nSedReturn: {} \nSuccess: {} \nTime: {} \n================\n".format(
                        ep+1, 
                        st+1, 
                        ep_rew,
                        sedimentary_returns[-1],
                        success_counter,
                        time.time()-start_time
                    )
                )
                # Update actor critic
                if not episode_counter%update_every:
                    # pdb.set_trace()
                    actor_dataset, critic_dataset = buf.get()
                    agent.train(actor_dataset, critic_dataset, num_epochs=80)
                    buf.__init__(size=buffer_size)
                break
                
    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()

    # save model
    if not os.path.exists(model_dir_actor):
        os.makedirs(model_dir_actor)
    if not os.path.exists(model_dir_critic):
        os.makedirs(model_dir_critic)
    tf.saved_model.save(agent.actor, model_dir_actor)
    tf.saved_model.save(agent.critic, model_dir_critic)
    
