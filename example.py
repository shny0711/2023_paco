import gym
import robel
from utils.wrapper import RecordEnv, RecordInfoEnv, InfoEnv
import wandb

# env = RecordInfoEnv(env)

wandb.init(entity="gyuta", project="ada-paco_test")
videopath = "e.mp4"
for ep in range(4):
    env = gym.make("MarathonFric0.001-v0")
    env = InfoEnv(env)
    if ep % 2 == 0:
        env = RecordInfoEnv(env, camera_id=1)
        env.start(videopath)
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        # env.write([str(obs), str(reward), str(done)])
        print(info)
        wandb.log({"reward": reward})
    if ep % 2 == 0:
        env.stop()

wandb.log({"movie": wandb.Video(videopath, fps=50.0, format="mp4")})