import argparse
from collections import OrderedDict
import gym
import robel
import wandb
import numpy as np
from algo.sac import SAC, MLPSAC , MHL_SAC_T
from utils.high_wrapper import HighRecordInfoEnv, HighInfoEnv
from utils.func import nowstr, get_strdate
from tqdm import tqdm
import itertools
from gym.spaces import Box
import matplotlib.pyplot as plt


#3次元
#たかさごと

class HighEnv(HighInfoEnv, MHL_SAC_T):
    def __init__(self, env):
        super().__init__(env)
        #以下いらん？
        self.action_space = Box(low=np.full(3, -1), high=np.full(3, 1), shape=None,  dtype=np.float32)

    # def exploits(self, state, sacs, high_action):
    #     low_actions = [sac.exploit(state) for sac in sacs]
    #     low_action = high_action * low_actions[0] +(1-high_action)*low_actions[1]
    #     return low_action


    def step(self, state, sacs, high_action):
        #low_action = sacs.exploits(high_action) # low_actionは１２次元

        low_action = hl_sac.get_lowaction(state, sacs, high_action)


        n_obs, reward, done, info = super().step(low_action)
        return n_obs, reward, done, info

def hl_train():
    best_ret = -float("inf")
    obs = high_env.reset()
    with tqdm(range(args.all_step+1)) as pbar:
        for t in pbar:
            high_action = hl_sac.explore(obs, only_action=True) if t > hl_sac.start_steps else high_env.action_space.sample()# high_action は１次元で重みを表している。
            n_obs, reward, done, info = high_env.step(obs, sacs, high_action)
            fall = done and not info["truncated"]
            hl_sac.hiact = high_action

            # wandb.log({"retrun":info["return"], "dist": info["obs/root_pos"][1]})

            hl_sac.buffer.append(obs, high_action, reward, fall, n_obs)
            obs=n_obs

            if done:
                ret = info["return"]
                pbar.set_postfix(OrderedDict(ret = ret))
                wandb.log({"train/return": ret})
                obs = high_env.reset()

            if hl_sac.is_update(t):
                hl_sac.update()

            if t % args.test_interval == 0:
                ret = test(t)
                
                if best_ret < ret:
                    best_ret = ret
                    hl_sac.save(f"{savepath}/best", {"return": float(best_ret), "t": t})
                    record(t)
            
            if args.record_interval > 0 and t % args.record_interval == 0:
                record(t)
            
            if args.save_num > 0 and t % (args.all_step // args.save_num) == 0:
                hl_sac.save(f"{savepath}/{t}", {"return": float(ret), "t": t})


def test(t):
    env = gym.make(args.env)
    env = HighEnv(env)
    returns = []
    steps = []
    pred_scores = []
    for i in range(args.test_num):
        done = False
        obs = env.reset()
        pred_score = 0
        while not done:
            #high_action = hl_sac.exploits(sacs, obs)
            high_action = hl_sac.exploit(obs)
            nobs, reward, done, info = env.step(obs, sacs, high_action)
            pred_score += hl_sac.pred_score(obs, high_action, nobs)
            obs = nobs
        returns.append(info["return"])
        steps.append(info["step"])
        pred_scores.append(pred_score)
    d = {f"test/{target[0]}-{method[0]}": method[1](target[1]) for target, method in itertools.product([("returns", returns), ("steps", steps), ("pred_scores", pred_scores)], [("mean", np.mean), ("std", np.std)])}
    d.update({"t":t})
    wandb.log(d)
    return np.mean(returns)

def record(t):
    env = gym.make(args.env)
    env = HighEnv(env)
    env = HighRecordInfoEnv(env, hl_sac, camera_id=camera_id)
    videopath = "video.mp4"
    env.start(videopath)

    for i in range(args.record_num):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            #high_action = hl_sac.exploits(obs)
            high_action = hl_sac.exploit(obs)
            n_obs, reward, done, info = env.step(obs, sacs, high_action)
            pred_score = hl_sac.pred_score(obs, high_action, n_obs)

            
            for k in range(3):
                wandb.log({f"h-{k}":high_action[k],f"tmp/w-{t}-{k}":(1+high_action[k])/2, "stepnum": step})


            env.write([
                nowstr(),
                __file__,
                args.env,
                f"t: {t}",
                str(info["step"]),
                str(info["return"]),
                f"score: {info['obs/root_pos'][1]}",
                f"pred score: {pred_score}"
            ]+ [
                f"base friction: {info.get('base_friction', 'None')}",
                f"high_action: {high_action}" ,
                f"weight of 1.0: {(1+high_action)/2}"
            ])
            obs = n_obs
            step += 1
    env.stop()
    wandb.log({"t": t, "retrun":info["return"], "score": info["obs/root_pos"][1],"record": wandb.Video(videopath, fps=50.0, format="mp4")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", default="MHL_SAC_T", choices=["SAC", "MLPSAC", "MHL_SAC_T"])
    parser.add_argument("--all-step", default=10**6, type=int)
    parser.add_argument("--env", default="MarathonFric0.1-v0")
    parser.add_argument("--test-interval", default=10**3, type=int)
    parser.add_argument("--test-num", default=1, type=int)
    parser.add_argument("--record-interval", default=10**4, type=int)
    parser.add_argument("--record-num", default=1, type=int)
    parser.add_argument("--save-num", default=-1, type=int)
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--replay-size", default=10**6, type=int)
    parser.add_argument("--start-step", default=10**3, type=int)
    #parser.add_argument("--path", default=['data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best', 'data/MarathonFric0.01-v0/MLPSAC/20231201_162424/best'])
    
    parser.add_argument("--path1")
    parser.add_argument("--path2")

    args = parser.parse_args()
    wandb.init(entity="kai-shunya", project="mhl_sac_3次元_data", config=args)

    camera_id = None if args.env in ["Pendulum-v0"] else 1
    date = get_strdate()
    savepath = f"data/{args.env}/{args.cls}/{date}"

    env = gym.make(args.env)
    high_env = HighEnv(env)

    pathes = [args.path1, args.path2]

    cls = eval(args.cls) #cls = HL_SAC
    hl_sac = cls(high_env.observation_space.shape, high_env.action_space.shape, device=args.device, alpha=args.alpha,
        batch_size=args.batch_size, replay_size=args.replay_size, start_steps=args.start_step)
    
    sacs =  [MLPSAC.load(path) for path in pathes]
    
    hl_train()
