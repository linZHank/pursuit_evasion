import sys
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


from envs.pe_env_1v1_discrete import PursuitEvasionOneVsOneDiscrete
from agents.dqn import DQNAgent


if __name__=='__main__':
    # instantiate env
    env = PursuitEvasionOneVsOneDiscrete(resolution=(150,150))
    # parameter
    num_episodes = 16
    num_steps = env.max_episode_steps
    # variables
    step_counter = 0
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    # instantiate agent
    agent_p = DQNAgent(buffer_size=int(1e4), warmup_episodes=5, batch_size=256, decay_period=5)
    start_time = time.time()
    for ep in range(num_episodes):
        obs, ep_rew = env.reset(), 0
        img = obs.copy()
        odom = np.concatenate((env.pursuers['position'][0],env.pursuers['velocity'][0]), axis=-1)
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
            # store transition
            agent_p.replay_buffer.store(img, odom, act[1], rew[1], done[1], nxt_img, nxt_odom)
            # train one step
            if ep >= agent_p.warmup_episodes:
                agent_p.train_one_step()
            # finish step
            ep_rew += rew[1]
            img = nxt_img.copy()
            odom = nxt_odom.copy()
            if info:
                episodic_returns.append(ep_rew)
                sedimentary_returns.append(sum(episodic_returns)/(ep+1))
                if done[0]:
                    success_counter += 1
                logging.info(
                    "\n================\nEpisode: {} \nEpisodeLength: {} \nEpisodeTotalRewards: {} \nAveragedTotalReward: {} \nSuccess: {} \nTime: {}".format(
                        ep+1, 
                        st+1, 
                        ep_rew,
                        sedimentary_returns[-1],
                        success_counter,
                        time.time()-start_time
                    )
                )
                break
                

