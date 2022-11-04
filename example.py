import gym
import robel
from utils.wrapper import RecordEnv, RecordInfoEnv

env = gym.make("DKittyWalkRandom-v0")
env = RecordInfoEnv(env)

env.start("d.mp4")
for i in range(1):
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.write([str(obs), str(reward), str(done)])
env.stop()