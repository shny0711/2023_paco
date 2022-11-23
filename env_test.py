import robel
import gym
from mujoco_py.generated import const
import numpy as np
from algo.sac import MLPSAC

agent = MLPSAC.load("data/MarathonFric1.0-v0/MLPSAC/20221116_123558/best")
env = gym.make("FollowFric1.0-v0")
for ep in range(10):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        act = agent.exploit(obs)
        obs, reward, done, info = env.step(act)

        # del v._markers[:]
