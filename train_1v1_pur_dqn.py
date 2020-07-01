import sys
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


from envs.pe_1v1_discrete import PursuitEvasionOneVsOneDiscrete
from agents.dqn import DQNAgent


if __name__=='__main__':
    # instantiate env
    env = PursuitEvasionOneVsOneDiscrete()
    # parameter
    num_episodes = 2000
    num_steps = env.max_episode_steps
    train_freq = 80
    # variables
    step_counter = 0
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    # instantiate agent
    agent_p = DQNAgent(name='dqn_pursuer')
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent_p.name, 'models')
    start_time = time.time()
    for ep in range(num_episodes):
        obs, ep_rew = env.reset(), 0
        img = obs.copy()
        odom = np.concatenate((env.pursuers['position'][0],env.pursuers['velocity'][0]), axis=-1)
        agent_p.linear_epsilon_decay(curr_ep=ep)
        for st in range(num_steps):
            # render for debug purpose
            # env.render(pause=1./env.rate)
            # cv2.imshow('map', img[:,:,[2,1,0]])
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
            act = np.array([0,0])
            act[1] = agent_p.epsilon_greedy(img, odom)
            obs, rew, done, info = env.step(act)
            nxt_img = obs.copy()
            nxt_odom = np.concatenate((env.pursuers['position'][0],env.pursuers['velocity'][0]), axis=-1) 
            ep_rew += rew[1]
            step_counter += 1
            # store transition
            agent_p.replay_buffer.store(img, odom, act[1], rew[1], done[1], nxt_img, nxt_odom)
            # train one step
            if ep >= agent_p.warmup_episodes:
                if not step_counter%train_freq:
                    for _ in range(train_freq):
                        agent_p.train_one_step()
            # finish step, EXTREMELY IMPORTANT!!!
            img = nxt_img.copy()
            odom = nxt_odom.copy()
            if info:
                episodic_returns.append(ep_rew)
                sedimentary_returns.append(sum(episodic_returns)/(ep+1))
                if done[0]:
                    success_counter += 1
                logging.info(
                    "\n================\nEpisode: {} \nEpsilon: {} \nEpisodeLength: {} \nEpisodeTotalRewards: {} \nAveragedTotalReward: {} \nSuccess: {} \nTime: {} \n================\n".format(
                        ep+1, 
                        agent_p.epsilon,
                        st+1, 
                        ep_rew,
                        sedimentary_returns[-1],
                        success_counter,
                        time.time()-start_time
                    )
                )
                break
                
    # save model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    agent_p.dqn_active.save_model(model_dir)
    
    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()
