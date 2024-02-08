import argparse
from algo.sac import MOESACS
import gym
import robel
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
#import pandas as pd
from utils.wrapper import RecordInfoEnv, InfoEnv

def main():
    reward_list=[]
    x_list=[]
    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    video = cv2. VideoWriter("switch.mp4", fourcc, 50.0, (1120, 480))
    t=0
    for i in range(1):
        obs = env.reset()
        done = False
        while not done:
            act = sacs.newmix(obs) #ここをかえる？
            n_obs, reward, done, info = env.step(act)
            sacs.update(obs, act, n_obs)
            obs = n_obs

            rgb = env.render(mode='rgb_array', camera_id=1)
            assert rgb.shape == (480, 480, 3)
            rgbcv = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            fig, ax = plt.subplots(2,2)
            ax[0,0].set_title(args.env)
            ax[0,1].set_title("return")
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

            

            # wandb.log({"t":t, "predict_error1.0":sacs.history[args.path1][t], "predict_error0.01":sacs.history[args.path2][t], "weight1.0":sacs.w_history[args.path1][t], "weight0.01":sacs.w_history[args.path2][t], "return":info["return"], "x":obs[1]})

            wandb.log({"t":t, "predict_error0.01":sacs.history[args.path1][t],"return":info["return"], "x":obs[1]})

            # wandb.log({"t":t, "return":info["return"], "x":obs[1]})

            t=t+1

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
            
            # figure=fig
            plt.close()

        # wandb.log({"fig":figure})

    #score_data = pd.DataFrame(sacs.history)
    #score_data.to_csv("scoredata/score_0.01.csv")
    video.release()

    wandb.log({"record": wandb.Video("switch.mp4", fps=50.0, format="mp4")})            


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", nargs="*", required=True)
    #parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best'])
    #parser.add_argument("--path", default=['data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    # parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best', 'data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    #parser.add_argument("--path", default=['data/MarathonFric0.1-v0/MLPSAC/20231205_122503/best'])

    # parser.add_argument("--path", default=["/home/shunya/dev/2023_11_paco/data/MarathonFric1.0-v0/MLPSAC/20240118_134921/best","/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_085151/best"])

    parser.add_argument("--path1", default="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_212046/best")
    parser.add_argument("--path2")

    parser.add_argument("--env", default="MarathonFric0.01-v0")
    parser.add_argument("--alpha", default=0.8, type=float)
    args = parser.parse_args()
    wandb.init(entity='kai-shunya', project='moe_data_single1.0', config=args)

    path = [args.path1, args.path2]

    path=[args.path1]

    sacs = MOESACS.load(path, alpha=args.alpha)
    
    env = gym.make(args.env)
    env = InfoEnv(env)
    main()