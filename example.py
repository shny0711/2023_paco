import robel
import gym

env = gym.make("DKittyWalkRandom-v0")

for i in range(10):
    env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())