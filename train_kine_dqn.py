#!/usr/bin/python3
"""
Training a single pursuers behavior using multiple instances
"""
from env.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils
import numpy as np
# from numpy import random


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    agent_p = DQNAgent(env=env)
    # train parameters
    num_episodes = 1000
    num_steps = env.max_steps
    num_epochs = 2
    episodic_returns = []
    sedimentary_returns = []
    ep = 0
    # train
    while ep < num_episodes:
        done, total_reward = False, []
        obs, _ = env.reset()
        state = dqn_utils.obs_to_state(obs)
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(3*num_episodes/5))
        for st in range(num_steps):
            action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            obs, val, done, _ = env.step(action_evaders, action_pursuers)
            next_states = dqn_utils.obs_to_state(obs)
            rew, done = agent_p.adjust_reward(obs, val)
            # store transitions
            agent_p.replay_memory.store(states, i_ap, rew, done, next_states)
            # train K epochs
            for i in range(num_epochs):
                agent_p.train()
            if not (ep+1)*(st+1) % agent_p.update_step:
                agent_p.qnet_stable.set_weights(agent_p.qnet_active.get_weights())
            # step summary
            print("episode: {}, step: {}, epsilon: {} \nstates: {} \naction: {}->{} \nnext_states: {} \nreward: {}".format(ep+1, st+1, agent_p.epsilon, states, i_ap, action_pursuers, next_states, rew))
            env.render()
            total_reward.append(rew)
            if done:
                break
        # episode summary
        episodic_returns.append(sum(total_reward))
        agent_p.save_model(model_dir)
        ep += 1
        print("episode: {} \n---\nepisodic_return: {}".format(ep+1, total_reward))
