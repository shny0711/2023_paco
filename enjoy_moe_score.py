import argparse
from algo.sac import MOESACS_SCORE
import gym
import robel
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
import pandas as pd

def main():
    reward_list=[]
    x_list=[]

    score1 = pd.read_csv("scoredata/score_1.0.csv")
    score2 = pd.read_csv("scoredata/score_0.01.csv")
    t=0

    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    video = cv2. VideoWriter("switch.mp4", fourcc, 50.0, (1120, 480))
    for i in range(1):
        obs = env.reset()
        done = False
        while not done:
            s1=score1.at[t, args.path1]
            s2=score2.at[t, args.path2]
            s_list = [s1,s2]
            t=t+1
            act = sacs.s_newmix(obs, s_list) 
            n_obs, reward, done, info = env.step(act)
            sacs.update(obs, act, n_obs)
            obs = n_obs

            rgb = env.render(mode='rgb_array', camera_id=1)
            assert rgb.shape == (480, 480, 3)
            rgbcv = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            fig, ax = plt.subplots(2,2)
            ax[0,0].set_title(args.env)
            ax[0,1].set_title("reward")
            ax[1,0].set_title("x")
            ax[1,1].set_title("weight")

            for name, h in sacs.history.items():
                ax[0,0].plot(h, label=name)

            for name, h in sacs.w_history.items():
                ax[1,1].plot(h, label=name)
            
            reward_list.append(info["reward/total"])
            ax[0,1].plot(reward_list)
            x_list.append(obs[1])
            ax[1,0].plot(x_list)

            wandb.log(sacs.history)
            wandb.log(info)
            ax[0,0].legend()
            ax[1,1].legend()
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

    #score_data = pd.DataFrame(sacs.history)
    #score_data.to_csv("scoredata/score_0.01.csv")
    video.release()
    wandb.log({"record": wandb.Video("switch.mp4", fps=50.0, format="mp4")})            


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", nargs="*", required=True)
    parser.add_argument("--path1", default='data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best')
    parser.add_argument("--path2", default='data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best')
    parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best', 'data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    #parser.add_argument("--path", default=['data/MarathonFric0.1-v0/MLPSAC/20231205_122503/best'])
    parser.add_argument("--env", default="MarathonFric0.01-v0")
    parser.add_argument("--alpha", default=0.8, type=float)
    args = parser.parse_args()
    wandb.init(entity='kai-shunya', project='new_moesacs', config=args)

    sacs = MOESACS_SCORE.load(args.path, alpha=args.alpha)
    
    env = gym.make(args.env)
    main()