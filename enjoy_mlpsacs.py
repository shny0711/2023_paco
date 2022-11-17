import argparse
from algo.sac import SACS
import gym
import robel
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    video = cv2.VideoWriter("switch.mp4", fourcc, 50.0, (1120, 480))
    for i in range(1):
        obs = env.reset()
        done = False
        while not done:
            act = sacs.exploit(obs)
            n_obs, reward, done, info = env.step(act)
            sacs.update(obs, act, n_obs)

            rgb = env.render(mode='rgb_array', camera_id=1)
            assert rgb.shape == (480, 480, 3)
            fig, ax = plt.subplots()
            for name, h in sacs.history.items():
                ax.plot(h, label=name)
            ax.legend()
            fig.canvas.draw()
            im = np.array(fig.canvas.renderer.buffer_rgba())
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            assert im.shape == (480, 640, 3)
            frame = np.concatenate((rgb,im),axis=1)
            assert frame.shape == (480, 1120, 3)
            video.write(frame)
            plt.close()
    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="*", required=True)
    parser.add_argument("--env", default="DKittyWalkFixed-v0")
    args = parser.parse_args()

    sacs = SACS.load(args.path)
    
    env = gym.make(args.env)
    main()