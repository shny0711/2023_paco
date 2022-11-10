import argparse
from algo.sac import SAC
import gym
import robel

def show():
    env = gym.make(args.env)
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(sac.exploit(obs))
        print(ep, info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--env", default="DKittyWalkFixed-v0")
    args = parser.parse_args()

    sac = SAC.load(args.path)
