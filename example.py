import gym
import robel
from utils.wrapper import RecordEnv, RecordInfoEnv
import wandb

env = gym.make("DKittyWalkRandom-v0")
env = RecordInfoEnv(env)

wandb.init(entity="gyuta", project="ada-paco_test")
videopath = "e.mp4"
env.start(videopath)
for i in range(1):
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.write([str(obs), str(reward), str(done)])
        wandb.log({"reward": reward})
env.stop()

wandb.log({"movie": wandb.Video(videopath, fps=50.0, format="mp4")})