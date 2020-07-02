import numpy as np
import cv2
from pe import PursuitEvasion
if __name__ == '__main__':
    env=PursuitEvasion()
    for ep in range(1):
        obs = env.reset()
        for st in range(env.max_episode_steps):
            env.render(pause=1./2)
            actions = np.random.uniform(-4,4,size=(env.num_evaders+env.num_pursuers,2))
            obs, rew, done, info = env.step(actions)
            img = obs[:,:,[2,1,0]]
            cv2.imshow('map', cv2.resize(img, (360, 360)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            print("\nevaders_pos: {} \npursuers_pos: {} \nreward: {} \ndone: {}".format(env.evaders['position'], env.pursuers['position'], rew, done))
            if info:
                print(info)
                break
    cv2.destroyAllWindows()

