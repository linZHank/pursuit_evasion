import numpy as np
import cv2
from pe import PursuitEvasion
if __name__ == '__main__':
    env=PursuitEvasion()
    for ep in range(10):
        obs = env.reset()
        # cv2.imshow('map', cv2.resize(env.image[:,:,[2,1,0]], (640, 640)))
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        # env.render(pause=2)
        for st in range(env.max_episode_steps):
            actions = np.random.uniform(-4,4,size=(env.num_evaders+env.num_pursuers,2))
            obs, rew, done, info = env.step(actions)
            cv2.imshow('map', cv2.resize(env.image[:,:,[2,1,0]], (640, 640)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            env.render(1./env.rate)
            print("\nevaders_pos: {} \npursuers_pos: {} \nreward: {} \ndone: {}".format(env.evaders['position'], env.pursuers['position'], rew, done))
            if info:
                print(info)
                break
