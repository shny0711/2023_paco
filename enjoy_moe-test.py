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
    data={
        "weight":[],
        "reward":[],
        "x":[],
    }

    with tqdm(range(1000)) as pber:
        for k in pber:
            wei_test=k/1000
            obs=env.reset()
            done=False
            while not done:
                act=sacs.mix_test(obs, wei_test)
                n_obs, reward, done, info = env.step(act)
                obs=n_obs

                
            data["weight"].append(wei_test)
            data["reward"].append(info["return"])
            data["x"].append(obs[1])

            wandb.log(data)

    df = pd.DataFrame(data)
    df.to_csv(args.savepath)


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", nargs="*", required=True)
    # parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best', 'data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    #parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best'])
    #parser.add_argument("--path", default=['data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    #parser.add_argument("--path", default=['data/MarathonFric0.1-v0/MLPSAC/20231205_122503/best'])

    parser.add_argument("--path1")
    parser.add_argument("--path2")
    parser.add_argument("--path", default=[]) 

    parser.add_argument("--savepath", default="testdata/weight_rewardpulus.csv")
    parser.add_argument("--env", default="MarathonFric0.1-v0")
    parser.add_argument("--alpha", default=0.8, type=float)
    args = parser.parse_args()
    wandb.init(entity='kai-shunya', project='paco_moe_test', config=args)

    args.path = [args.path1, args.path2]

    sacs = MOESACS_TEST.load(args.path, alpha=args.alpha)
    
    env = gym.make(args.env)
    env = InfoEnv(env)
    main()