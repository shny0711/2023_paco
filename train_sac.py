import argparse
from collections import OrderedDict
import gym
import robel
import wandb
import numpy as np
from algo.sac import SAC
from utils.wrapper import RecordInfoEnv, InfoEnv
from utils.func import nowstr
from tqdm import tqdm

def train():
    best_ret = -float("inf")
    obs = env.reset()
    with tqdm(range(args.all_step)) as pbar:
        for t in pbar:
            act = sac.explore(obs, only_action=True) if t > sac.start_steps else env.action_space.sample()
            n_obs, reward, done, info = env.step(act)
            fall = done and not info["truncated"]
            sac.buffer.append(obs, act, reward, fall, n_obs)
            obs = n_obs

            if done:
                ret = info["return"]
                pbar.set_postfix(OrderedDict(ret = ret))
                wandb.log({"train/return": ret})
                obs = env.reset()

            if sac.is_update(t):
                sac.update()

            if t % args.test_interval == 0:
                ret = test(t)
                
                if best_ret < ret:
                    best_ret = ret
                    record(t)
            
            if args.record_interval > 0 and t % args.record_interval == 0:
                record(t)

def test(t):
    env = gym.make(args.env)
    env = InfoEnv(env)
    rets = []
    steps = []
    for i in range(args.test_num):
        done = False
        obs = env.reset()
        while not done:
            act = sac.exploit(obs)
            obs, reward, done, info = env.step(act)
        rets.append(info["return"])
        steps.append(info["step"])
    ret, std = np.mean(rets), np.std(rets)
    step, step_std = np.mean(steps), np.std(steps)
    wandb.log({"t":t, "test/return-mean": ret, "test/return-std": std, "test/step-mean": step, "test/step-std": step_std})
    return ret

def record(t):
    env = gym.make(args.env)
    env = InfoEnv(env)
    env = RecordInfoEnv(env, camera_id=camera_id)
    videopath = "video.mp4"
    env.start(videopath)
    for i in range(args.record_num):
        obs = env.reset()
        done = False
        while not done:
            act = sac.exploit(obs)
            obs, reward, done, info = env.step(act)
            env.write([
                nowstr(),
                __file__,
                args.env,
                f"t: {t}",
                str(info["step"]),
                str(info["return"])
            ])
    env.stop()
    wandb.log({"t": t, "record": wandb.Video(videopath, fps=50.0, format="mp4")})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-step", default=10**6, type=int)
    parser.add_argument("--env", default="DKittyWalkFixed-v0")
    parser.add_argument("--test-interval", default=10**4, type=int)
    parser.add_argument("--test-num", default=1, type=int)
    parser.add_argument("--record-interval", default=0, type=int)
    parser.add_argument("--record-num", default=1, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--replay-size", default=10**6, type=int)
    args = parser.parse_args()
    wandb.init(entity="gyuta", project="paco_sac", config=args)

    camera_id = None if args.env in ["Pendulum-v0"] else 1

    env = gym.make(args.env)
    env = InfoEnv(env)
    sac = SAC(env.observation_space.shape, env.action_space.shape, device=args.device, alpha=args.alpha,
        batch_size=args.batch_size, replay_size=args.replay_size)
    train()