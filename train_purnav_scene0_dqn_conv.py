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

from envs.purnav_scene0_discrete import PursuerNavigationScene0Discrete
from agents.dqn_conv import ReplayBuffer, DeepQNet


if __name__=='__main__':
    env = PursuerNavigationScene0Discrete() # default resolution:(80,80)
    agent = DeepQNet(
        dim_obs=env.observation_space_shape, 
        num_act=env.action_options.shape[0], 
    ) 
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent.name)
    # params
    max_episode_steps = env.max_episode_steps
    batch_size = 32
    update_freq = 100
    update_after = 10000
    warmup_episodes = 200
    decay_period = 1000
    replay_buffer = ReplayBuffer(dim_obs=agent.dim_obs, size=int(2e5)) 
    total_steps = int(1e6)
    episodic_returns = []
    sedimentary_returns = []
    episodic_steps = []
    save_freq = 100
    episode_counter = 0
    success_counter = 0
    obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
    # state = np.stack([np.sum(obs, axis=-1) for _ in range(agent.dim_obs[-1])], axis=2)
    start_time = time.time()
    for t in range(total_steps):
        # env.render()
        act = np.squeeze(agent.act(np.expand_dims(obs, axis=0)))
        n_obs, rew, done, info = env.step(int(act))
        ep_ret += rew
        # # next 4 lines update stack observation
        # n_state = state.copy()
        # for i in range(state.shape[-1]-1):
        #     n_state[:,:,i] = n_state[:,:,i+1]
        # n_state[:,:,-1] = np.sum(obs, axis=-1)
        ep_len += 1
        done = False if ep_len == max_episode_steps else done
        replay_buffer.store(obs, act, rew, done, n_obs)
        obs = n_obs.copy() # SUPER CRITICAL
        if done or (ep_len==max_episode_steps):
            episode_counter += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/episode_counter)
            episodic_steps.append(t+1)
            if info == 'Goal reached!':
                success_counter += 1
            print("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpisodeReturn: {} \nSucceeded: {} \nSedimentaryReturn: {} \nTimeElapsed: {} \n====\n".format(episode_counter, ep_len, t+1, ep_ret, success_counter, sedimentary_returns[-1], time.time()-start_time))
            # reset env
            obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
            agent.linear_epsilon_decay(episode_counter, decay_period, warmup_episodes)
            # state = np.stack([np.sum(obs, axis=-1) for _ in range(agent.dim_obs[-1])], axis=2)
            # save model
            if not episode_counter%save_freq:
                model_path = os.path.join(model_dir, str(episode_counter))
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                agent.q.q_net.save(model_path)
        if not t%update_freq and t>=update_after:
            for _ in range(update_freq):
                minibatch = replay_buffer.sample_batch(batch_size=batch_size)
                loss_q = agent.train_one_batch(data=minibatch)
                logging.debug("\nloss_q: {}".format(loss_q))

    # Save returns 
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))
    # Save final model
    model_path = os.path.join(model_dir, str(episode_counter))
    agent.q.q_net.save(model_path)

    # Test
    input("Press ENTER to test agent...")
    for ep in range(10):
        o, d, ep_ret = env.reset(), False, 0
        for st in range(env.max_episode_steps):
            cv2.imshow('map', cv2.resize(obs[:,:,[2,1,0]], (640, 640)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            env.render(pause=1./env.rate)
            a = np.squeeze(agent.act(np.expand_dims(o, axis=0)))
            o2,r,d,_ = env.step(int(a))
            ep_ret += r
            o = o2
            if d:
                print("EpReturn: {}".format(ep_ret))
                break 
    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()
    cv2.destroyAllWindows()

