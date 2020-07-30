import numpy as np
import cv2
from pur_nav import PursuerNavigation
if __name__ == '__main__':
    env=PursuerNavigation()
    for ep in range(10):
        obs = env.reset()
        for st in range(env.max_episode_steps):
            cv2.imshow('map', cv2.resize(obs[:,:,[2,1,0]], (640, 640)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            env.render(pause=1./env.rate)
            action = np.random.uniform(-4,4,size=2)
            obs, rew, done, info = env.step(action)
            print("\nepisode: {} \nstep: {} \nevaders_pos: {} \npursuers_pos: {} \nreward: {} \ndone: {}".format(ep+1, st+1, env.evaders['position'], env.pursuers['position'], rew, done))
            if info:
                print(info)
                break
    cv2.destroyAllWindows()
