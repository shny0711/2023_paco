import argparse
from algo.sac import SAC, MLPSAC , HL_SAC
import gym
import robel
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
#import pandas as pd
from utils.high_wrapper import HighRecordInfoEnv, HighInfoEnv
from gym.spaces import Box

class HighEnv(HighInfoEnv, HL_SAC):
    def __init__(self, env):
        super().__init__(env)
        #以下いらん？
        self.action_space = Box(low=np.array([-1]), high=np.array([1]), shape=None,  dtype=np.float32)


    def step(self, state, sacs, high_action):
        #low_action = sacs.exploits(high_action) # low_actionは１２次元

        low_action = hl_sac.get_lowaction(state, sacs, high_action)


        n_obs, reward, done, info = super().step(low_action)
        return n_obs, reward, done, info

def main():
    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    video = cv2. VideoWriter("switch.mp4", fourcc, 50.0, (1120, 480))
    t=0
    for i in range(1):
        obs = high_env.reset()
        done = False
        w_list=[]
        while not done:
            high_action = hl_sac.exploit(obs)
            n_obs, reward, done, info = high_env.step(obs, sacs, high_action)
            obs = n_obs

            rgb = env.render(mode='rgb_array', camera_id=1)
            assert rgb.shape == (480, 480, 3)
            rgbcv = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            fig, ax = plt.subplots()
            w_list.append((1+high_action)/2)
            ax.set_title("weight")
            ax.plot(w_list)
            wandb.log({"t":t, "return":info["return"], "x":obs[1], "weight1.0":(1+high_action)/2})

            t=t+1

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
    parser.add_argument("--path", default="data/MarathonFric0.1-v0/HL_SAC/20240121_222744/best")
    parser.add_argument("--env", default="MarathonFric0.1-v0")
    parser.add_argument("--alpha", default=0.8, type=float)
    args = parser.parse_args()
    wandb.init(entity='kai-shunya', project='hl_data_single', config=args)


    hl_sac = HL_SAC.load(args.path)

    pathes=["/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240119_023028/best", "/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_212046/best"]
    sacs = [MLPSAC.load(path) for path in pathes]

    env = gym.make(args.env)
    high_env = HighEnv(env)
    main()