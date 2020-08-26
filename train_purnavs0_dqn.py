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
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from envs.purnavs0 import PursuerNavigationScene0
from agents.dqn import ReplayBuffer, DeepQNet


if __name__=='__main__':
    env = PursuerNavigationScene0() # default resolution:(80,80)
    agent = DeepQNet(
        dim_obs=[8], 
        num_act=env.action_reservoir.shape[0], 
        lr=3e-4
    ) 
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent.name)
    # params
    max_episode_steps = env.max_episode_steps
    batch_size = 128
    update_freq = 100
    update_after = 10000
    warmup_episodes = 500
    decay_period = 2000
    replay_buffer = ReplayBuffer(dim_obs=agent.dim_obs, size=int(1e6)) 
    total_steps = int(5e6)
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
        state = np.array([
            env.evaders['position'][0],
            env.evaders['velocity'][0],
            env.pursuers['position'][0],
            env.pursuers['velocity'][0],
        ]).reshape(-1)
        act = np.squeeze(agent.act(np.expand_dims(state, axis=0)))
        n_obs, rew, done, info = env.step(int(act))
        n_state = np.array([
            env.evaders['position'][0],
            env.evaders['velocity'][0],
            env.pursuers['position'][0],
            env.pursuers['velocity'][0],
        ]).reshape(-1)
        logging.debug("\nstate: {} \naction: {} \nreward: {} \ndone: {} \nn_state: {}".format(state, act, rew, done, n_state))
        ep_ret += rew
        # # next 4 lines update stack observation
        # n_state = state.copy()
        # for i in range(state.shape[-1]-1):
        #     n_state[:,:,i] = n_state[:,:,i+1]
        # n_state[:,:,-1] = np.sum(obs, axis=-1)
        ep_len += 1
        done = False if ep_len == max_episode_steps else done
        replay_buffer.store(state, act, rew, done, n_state)
        state = n_state.copy() # SUPER CRITICAL
        if done or (ep_len==max_episode_steps):
            episode_counter += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/episode_counter)
            episodic_steps.append(t+1)
            if info == 'Goal reached!':
                success_counter += 1
            logging.info("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpisodeReturn: {} \nSucceeded: {} \nSedimentaryReturn: {} \nTimeElapsed: {} \n====\n".format(episode_counter, ep_len, t+1, ep_ret, success_counter, sedimentary_returns[-1], time.time()-start_time))
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
                print("\nloss_q: {}".format(loss_q))

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
            s = np.array([
                env.evaders['position'][0],
                env.evaders['velocity'][0],
                env.pursuers['position'][0],
                env.pursuers['velocity'][0],
            ]).reshape(-1)
            a = np.squeeze(agent.act(np.expand_dims(s, axis=0)))
            o2,r,d,_ = env.step(int(a))
            s2 = np.array([
                env.evaders['position'][0],
                env.evaders['velocity'][0],
                env.pursuers['position'][0],
                env.pursuers['velocity'][0],
            ]).reshape(-1)
            ep_ret += r
            s = s2.copy()
            if d:
                print("EpReturn: {}".format(ep_ret))
                break 
    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()
    cv2.destroyAllWindows()


