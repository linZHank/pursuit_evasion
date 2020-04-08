import sys
import os
import numpy as np
from numpy import random
import pickle


def obs_to_states_solo(obs, num_pursuers):
    states = np.zeros((num_pursuers,8))
    for i in range(num_pursuers):
        states[i] = np.concatenate((obs[i*4:i*4+4], obs[-4:]))

    return states

def obs_to_states(obs, num_pursuers, num_evaders):
    """
    Convert env's raw observation into agent's state input
    Args:
        obs: array([x_p0,y_p0,...,vx_p0,vy_p0,...,x_e0,y_e0,...,vx_e0,vy_e0,...])
    Returns:
        state: array([[x_p0,y_p0,...], [x_p1,y_p1,...], ..., [x_en,y_en,...]])
    """
    states = np.tile(obs, (num_pursuers+num_evaders,1))
    # reorder pursuers
    for i in range(num_pursuers):
        s = obs
        s[[0,1,i*2,i*2+1]] = s[[i*2,i*2+1,0,1]] # swap pos
        s[[num_pursuers*2,num_pursuers*2+1,num_pursuers*2+i*2,num_pursuers*2+i*2+1]] = s[[num_pursuers*2+i*2,num_pursuers*2+i*2+1,num_pursuers*2,num_pursuers*2+1]] # swap vel
        states[i] = s
    # reorder evaders
    for i in range(num_evaders):
        s = obs
        s[[num_pursuers*4,num_pursuers*4+1,num_pursuers*4+i*2,num_pursuers*4+i*2+1]] = s[[num_pursuers*4+i*2,num_pursuers*4+i*2+1,num_pursuers*4,num_pursuers*4+1]] # swap pos
        s[[num_pursuers*4+num_evaders*2,num_pursuers*4+num_evaders*2+1,num_pursuers*4+num_evaders*2+i*2,num_pursuers*4+num_evaders*2+i*2+1]] = s[[num_pursuers*4+num_evaders*2+i*2,num_pursuers*4+num_evaders*2+i*2+1,num_pursuers*4+num_evaders*2,num_pursuers*4+num_evaders*2+1]] # swap vel
        states[num_pursuers+i] = s

    return states

def circular_action(pos, speed):
    """
    Generate action to enable agent moving in circle regarding to (0,0)
    Args:
        pos: array([x,y])
        speed: tangential velocity, +: counter-clockwise
    Returns:
        action: array([v_x,v_y])
    """
    rot = np.array([[0,-1], [1,0]]) # rotation matrix
    vec_vel = np.dot(rot, pos)
    norm_vec_vel = vec_vel/np.linalg.norm(vec_vel)
    action = speed*norm_vec_vel # .reshape(1,-1)

    return action

def adjust_reward(env, num_steps, state, reward, done, next_state):
    """
    Args:
        env: env object
        state: array([x_p,y_p,x_e,y_e])
    Returns:
        reward: scalar
        done: boolean
    """
    success = False
    if env.pursuers['status'][0] == 'occluded' or env.pursuers['status'][0] == 'out':
        reward = -1
        done = True
    elif env.pursuers['status'][0] == 'catching':
        reward = env.world_length
        done = True
        success = True
    else:
        if np.linalg.norm(state[:2]-state[-2:]) < np.linalg.norm(next_state[:2]-next_state[-2:]):
            reward = -1./num_steps
        else:
            reward = 0

    return reward, done, success

# save pickle
def save_pkl(content, fdir, fname):
    """
    Save content into path/name as pickle file
    Args:
        path: file path, str
        content: to be saved, array/dict/list...
        fname: file name, str
    """
    file_path = os.path.join(fdir, fname)
    with open(file_path, 'wb') as f:
        pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)
