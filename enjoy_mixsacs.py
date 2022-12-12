import argparse
from algo.sac import SACS
from algo.paco import MixSAC
import gym
import robel
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb

def main():
    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    video = cv2.VideoWriter("switch.mp4", fourcc, 50.0, (1120, 480))
    for i in range(1):
        obs = env.reset()
        done = False
        history = []
        while not done:
            act = mixsacs.main.exploit(obs)
            n_obs, reward, done, info = env.step(act)
            mixsacs.update(obs, act, n_obs)
            print(mixsacs.ws)
            obs = n_obs

            rgb = env.render(mode='rgb_array', camera_id=1)
            assert rgb.shape == (480, 480, 3)
            rgbcv = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            fig, ax = plt.subplots()
            ax.set_title(args.env)
            history.append(mixsacs.ws)
            for v, name in zip(zip(*history), args.path):
                ax.plot(v, label=name)
            ax.legend()
            fig.canvas.draw()
            im = np.array(fig.canvas.renderer.buffer_rgba())
            imcv = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            assert imcv.shape == (480, 640, 3)
            frame = np.concatenate((rgbcv,imcv),axis=1)
            assert frame.shape == (480, 1120, 3)
            video.write(frame)
            if done:
                wandb.log({"ws": im})
            plt.close()
    video.release()
    wandb.log({"record": wandb.Video("switch.mp4", fps=50.0, format="mp4")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="*", required=True)
    parser.add_argument("--env", default="MarathonC__H_ML-v0")
    parser.add_argument("--alpha", default=0.8, type=float)
    args = parser.parse_args()
    wandb.init(entity='gyuta', project='paco_mixsacs', config=args)

    mixsacs = MixSAC.load(args.path,  ws=[0.5, 0.5], sacs_kwarg={"alpha": 0})
    
    env = gym.make(args.env)
    main()