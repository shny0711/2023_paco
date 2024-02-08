import argparse
from algo.sac import SACS
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
        while not done:
            act = sacs.exploit(obs)
            n_obs, reward, done, info = env.step(act)
            sacs.update(obs, act, n_obs)
            obs = n_obs

            rgb = env.render(mode='rgb_array', camera_id=1)
            assert rgb.shape == (480, 480, 3)
            rgbcv = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            fig, ax = plt.subplots()
            ax.set_title(args.env)
            for name, h in sacs.history.items():
                ax.plot(h, label=name)
            wandb.log(sacs.history)
            ax.legend()
            fig.canvas.draw()
            im = np.array(fig.canvas.renderer.buffer_rgba())
            imcv = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            assert imcv.shape == (480, 640, 3)
            frame = np.concatenate((rgbcv,imcv),axis=1)
            assert frame.shape == (480, 1120, 3)
            video.write(frame)
            if done:
                wandb.log({"predict error": im})
            plt.close()
    video.release()
    wandb.log({"record": wandb.Video("switch.mp4", fps=50.0, format="mp4")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", nargs="*", required=True)
    parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best'])
    parser.add_argument("--env", default="MarathonC__H_ML-v0")
    parser.add_argument("--alpha", default=0.8, type=float)
    args = parser.parse_args()
    wandb.init(entity='kai-shunya', project='paco_mlpsacs', config=args)

    sacs = SACS.load(args.path, alpha=args.alpha)
    
    env = gym.make(args.env)
    main()