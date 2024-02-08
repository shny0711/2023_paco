import argparse
from algo.sac import MOESACS_TEST
import gym
import robel
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
import pandas as pd
from tqdm import tqdm
from utils.wrapper import RecordInfoEnv, InfoEnv

def main():
    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    video = cv2. VideoWriter("switch.mp4", fourcc, 50.0, (1120, 480))

    # pattern1 w=0.07 path_1(miss)
    # pattern2 w=0.07 path_3(success)
    # pattern3 w=0.4 path3
    # pattern4 w=0.8 path4

    k=args.weight
    k=int(k)
    wei_test = k/1000
    obs=env.reset()
    done=False
    while not done:
        act=sacs.mix_test(obs, wei_test)
        n_obs, reward, done, info = env.step(act)
        obs=n_obs

        rgb = env.render(mode='rgb_array', camera_id=1)
        assert rgb.shape == (480, 480, 3)
        rgbcv = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        fig, ax = plt.subplots()
        ax.set_title(args.env)
        ax.plot(k)
        fig.canvas.draw()

        im = np.array(fig.canvas.renderer.buffer_rgba())
        imcv = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        assert imcv.shape == (480, 640, 3)
        frame = np.concatenate((rgbcv,imcv),axis=1)
        assert frame.shape == (480, 1120, 3)
        video.write(frame)
        if done:
            wandb.log({"predict error": im})
                        # figure=fig
        plt.close()
    
    video.release()
    wandb.log({"record": wandb.Video("switch.mp4", fps=50.0, format="mp4"), "return":info["return"], "distance":obs[1]})    




if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", nargs="*", required=True)
    # parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best', 'data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    #parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best'])
    #parser.add_argument("--path", default=['data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    #parser.add_argument("--path", default=['data/MarathonFric0.1-v0/MLPSAC/20231205_122503/best'])

    parser.add_argument("--path1")
    parser.add_argument("--path2")
    parser.add_argument("--weight", default=400)

    parser.add_argument("--env", default="MarathonFric0.1-v0")
    parser.add_argument("--alpha", default=0.8, type=float)
    args = parser.parse_args()
    wandb.init(entity='kai-shunya', project='paco_moe_movie', config=args)

    path = [args.path1, args.path2]

    sacs = MOESACS_TEST.load(path, alpha=args.alpha)
    
    env = gym.make(args.env)
    env = InfoEnv(env)
    main()