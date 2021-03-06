import numpy as np
import cv2
from envs.pe1v1 import PursuitEvasionOneVsOne
if __name__ == '__main__':
    env=PursuitEvasionOneVsOne()
    for ep in range(10):
        o = env.reset()
        # cv2.imshow('map', cv2.resize(obs[:,:,[2,1,0]], (640, 640)))
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        # env.render(pause=2)
        for st in range(env.max_episode_steps):
            cv2.imshow('map', cv2.resize(o[:,:,[2,1,0]], (640, 640)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            env.render(pause=1./env.rate)
            a = np.random.randint(low=0, high=env.action_reservoir.shape[0], size=2)
            o, r, d, i = env.step(a)
            print("\nepisode: {} \nstep: {} \nevaders_pos: {} \npursuers_pos: {} \nreward: {} \ndone: {}".format(ep+1, st+1, env.evaders['position'], env.pursuers['position'], r, d))
            if i:
                print(i)
                break
    cv2.destroyAllWindows()
