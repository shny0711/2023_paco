from abc import ABC, abstractmethod
import os
import glob
from collections import deque
from time import time
from datetime import timedelta
import joblib
from base64 import b64encode
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from datetime import datetime
import japanize_matplotlib
import wandb
from utils.func import write_dict, load_dict
from collections import OrderedDict

from algo.filter import IdentityFilter

class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=3):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードを設定する．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': [], "heading":[], "dist":[]}

        # データ収集を行うステップ数．
        self.num_steps = num_steps
        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes

    def train(self, path):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        path = f"{path}/policy"
        os.mkdir(path)

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．

            state, t = self.algo.step(self.env, state, t, steps)

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                self.algo.update()

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0:
                self.evaluate(steps)
            
            if steps % (self.num_steps//10) == 0:
                self.save_policy(path, steps)

    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        returns = []
        headings = []
        dists = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, info = self.env_test.step(action)
                episode_return += reward

            returns.append(episode_return)
            try:
                headings.append(info["obs/heading"])
                x,y = info["obs/target_error"]
            except:
                headings.append(0)
                x,y = 0,0
            dist = math.sqrt(x**2 + y**2)
            dists.append(dist)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)
        self.returns["heading"].append(np.mean(headings))
        self.returns["dist"].append(np.mean(dists))

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}   '
              f'Now: {datetime.now().strftime("%H:%M:%S")}')

    def visualize(self):
        """ 1エピソード環境を動かし，mp4を再生する． """
        env = wrap_monitor(gym.make(self.env.unwrapped.spec.id))
        state = env.reset()
        done = False

        while (not done):
            action = self.algo.exploit(state)
            state, _, done, _ = env.step(action)

        del env
        return play_mp4()

    def plot(self):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.tight_layout()
    
    def save_policy(self, path, name):
        """ ポリシーを保存
        """
        actor = self.algo.actor
        torch.save(actor.state_dict(), f"{path}/{name}.pth")
    
    def save_record(self, path):
        with open(path+"/returns.joblib", "wb") as f:
            joblib.dump(self.returns, f)
    
    def save_graph(self, name, path):
        """ 平均収益のグラフを保存 """

        ds = [
            {"key": "return", "title": "評価時におけるリターンの変化"},
            {"key": "heading", "title": "評価時におけるエピソード終了時のゴールとの向きのずれの変化"},
            {"key": "dist", "title": "評価時におけるエピソード終了時のゴールまでの距離の変化"},
        ]
        for d in ds:
            k = d["key"]
            t = d["title"]

            fig, ax = plt.subplots()
            ax.set_xlabel("steps")
            ax.set_ylabel(k)
            ax.set_title(t)
            ax.plot(self.returns['step'], self.returns[k], label=name)
            ax.legend()
            plt.savefig(path+f"/{k}.png")

    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time() - self.start_time)))

class Algorithm(ABC):

    def explore(self, state, only_action=False):
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        if only_action:
            return action.cpu().numpy()[0]
        else:
            return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        """ 決定論的な行動を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, steps):
        """ 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す． """
        pass

    @abstractmethod
    def update(self):
        """ 1回分の学習を行う． """
        pass

def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    # (torch.distributions.Normalを使うと無駄な計算が生じるので，下記では直接計算しています．)
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # tanh による確率密度の変化を修正する．
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis

def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanh　を適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)

    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis

def atanh(x):
    """ tanh の逆関数． """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    """ 平均(mean)，標準偏差の対数(log_stds)でパラメータ化した方策における，行動(actions)の確率密度の対数を計算する． """
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

class SACActor(nn.Module):

    def __init__(self, state_shape, action_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * action_shape[0]),
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))

    def evaluate_log_pi(self, states, actions):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return evaluate_lop_pi(means, log_stds.clamp(-20, 2), actions)

class SACCritic(nn.Module):

    def __init__(self, state_shape, action_shape):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)

class ReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device):
        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )

class SAC(Algorithm):

    def __init__(self, state_shape, action_shape, device=torch.device('cpu'), seed=0,
                 batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0):
        super().__init__()

        # シードを設定する．
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

        # リプレイバッファ．
        self.buffer = ReplayBuffer(
            buffer_size=replay_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )

        # Actor-Criticのネットワークを構築する．
        self.actor = SACActor(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)
        self.critic = SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)
        self.critic_target = SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device).eval()

        # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

        self.device = device

        self.filter = IdentityFilter()

        self.config = {
            "state_shape": state_shape,
            "action_shape": action_shape,
            "batch_size": batch_size,
            "gamma": gamma,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "replay_size": replay_size,
            "start_steps": start_steps,
            "tau": tau,
            "alpha": alpha,
            "reward_scale": reward_scale
        }

    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= max(self.start_steps, self.batch_size)

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

        wandb.log({"train/critic1": loss_critic1, "train/critic2": loss_critic2})

    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()
        
        wandb.log({"train/actor": loss_actor})

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
    
    def pred_score(self, states, actions, next_states):
        return -1
    
    def save(self, path, info={}):
        os.makedirs(path, exist_ok=True)
        write_dict(f"{path}/info.yml", info)
        write_dict(f"{path}/config.yml", self.config)
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        torch.save(self.critic_target.state_dict(), f"{path}/critic_target.pth")
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")

    @classmethod
    def load(cls, path):
        config = load_dict(f"{path}/config.yml")
        #config = load_dict("data/MarathonFric1.0-v0/MLPSAC/20231130_165623/best/config.yml")
        state_shape = config.pop("state_shape")
        action_shape = config.pop("action_shape")
        sac = cls(state_shape, action_shape, **config)
        sac.critic.load_state_dict(torch.load(f"{path}/critic.pth"))
        sac.critic_target.load_state_dict(torch.load(f"{path}/critic_target.pth"))
        sac.actor.load_state_dict(torch.load(f"{path}/actor.pth"))

        return sac

class MLPSAC(SAC):
    def __init__(self, state_shape, action_shape, lr_optim=1e-3, **kwargs):
        super().__init__(state_shape, action_shape, **kwargs)
        self.predictor = nn.Sequential(
            nn.Linear(state_shape[0]+action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, state_shape[0]),
        ).to(self.device)
        self.pred_optim = torch.optim.Adam(self.predictor.parameters(), lr=lr_optim)
    
    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()
        self.update_predictor(states, actions, next_states)
    
    def update_predictor(self, states, actions, next_states):
        inp = torch.cat((states, actions), dim = 1)
        out = self.predictor(inp)
        loss = torch.nn.functional.mse_loss(out, next_states)
        self.pred_optim.zero_grad()
        loss.backward()
        self.pred_optim.step()

        wandb.log({"train/predictor": loss})

    def pred_score(self, states, actions, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        inp = torch.cat((states, actions), dim = 0)
        out = self.predictor(inp)
        loss = torch.nn.functional.mse_loss(out, next_states)
        return loss.item()
        
    def save(self, path, info={}):
        super().save(path, info)
        torch.save(self.predictor.state_dict(), f"{path}/predictor.pth")

    @classmethod
    def load(cls, path):
        sac = super().load(path)
        sac.predictor.load_state_dict(torch.load(f"{path}/predictor.pth"))

        return sac
    

class HL_SAC(SAC):
    def __init__(self, state_shape, action_shape, **kwargs):
        super().__init__(state_shape, action_shape, **kwargs)

    def get_lowaction(self, state, sacs, high_action):
        low_actions = [sac.exploit(state) for sac in sacs]
        low_action = (1+high_action) * low_actions[0]/2 +(1-high_action)*low_actions[1]/2
        return low_action

class MHL_SAC(SAC):
    def __init__(self, state_shape, action_shape, **kwargs):
        super().__init__(state_shape, action_shape, **kwargs)

    def get_lowaction(self, state, sacs, high_action):
        hact_array = np.array(high_action)
        low_actions = [np.array(sac.exploit(state)) for sac in sacs]

        low_action = (1+hact_array)*low_actions[0]/2 + (1-hact_array)*low_actions[1]/2
        return low_action

class MHL_SAC_T(SAC):
    def __init__(self, state_shape, action_shape, **kwargs):
        super().__init__(state_shape, action_shape, **kwargs)

    def get_lowaction(self, state, sacs, high_action):
        h = np.array(high_action)
        hact_array = np.concatenate([h,h,h,h],0)
        low_actions = [np.array(sac.exploit(state)) for sac in sacs]

        low_action = (1+hact_array)*low_actions[0]/2 + (1-hact_array)*low_actions[1]/2
        return low_action

class MHL_SAC_F(SAC):
    def __init__(self, state_shape, action_shape, **kwargs):
        super().__init__(state_shape, action_shape, **kwargs)

    def get_lowaction(self, state, sacs, high_action):
        h = np.array(high_action)
        hact_array = np.array([h[0],h[0],h[0], h[1],h[1],h[1], h[2],h[2],h[2], h[3],h[3],h[3]])
        low_actions = [np.array(sac.exploit(state)) for sac in sacs]

        low_action = (1+hact_array)*low_actions[0]/2 + (1-hact_array)*low_actions[1]/2
        return low_action


class SACS:
    """ MLPの予測誤差が最も低いSACを利用するやつ
    """
    def __init__(self, sacs, alpha=0.8, names=None, device=torch.device('cpu')):
        self.sacs = sacs
        self.scores = [0 for _ in sacs]
        self.alpha = alpha
        self.device = device

        if not names:
            self.names = [str(i) for i in range(len(sacs))]
        self.names = names
    
        self.history = OrderedDict([(name,[]) for name in names])

    def exploit(self, states):
        return self.best.exploit(states)

    @property
    def best(self):
        return self.sacs[self.scores.index(min(self.scores))]
    #[0,1]を選択している部分？

    def update(self, states, actions, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        inp = torch.cat((states, actions), dim = 0)
        outs = [sac.predictor(inp).cpu().detach().numpy() for sac in self.sacs]
        
        def mse(A,B):
            return ((A - B)**2).mean()
        
        def moving_ave(o, n):
            return self.alpha*o + (1-self.alpha)*n

        self.scores = [moving_ave(score, mse(out, next_states)) for score, out in zip(self.scores, outs)]
        for h, score in zip(self.history.values(), self.scores):
            h.append(score)

    @classmethod
    def load(cls, pathes, **kwargs):
        sacs = [MLPSAC.load(path) for path in pathes]
        return cls(sacs, names = pathes, **kwargs)
    




class MOESACS:
    """
    MLPの予測誤差を重み付けして行動を合成
    たぶんupdateはそのままでいい
    bestで選択しているから、そこをいじる？    
    
    """
    def __init__(self, sacs, alpha=0.8, names=None, device=torch.device('cpu')):
        self.sacs = sacs
        self.scores = [0 for _ in sacs]
        self.alpha = alpha
        self.device = device

        if not names:
            self.names = [str(i) for i in range(len(sacs))]
        self.names = names
    
        self.history = OrderedDict([(name,[]) for name in names])
        self.w_history = OrderedDict([(name,[]) for name in names])

    """def exploit(self, states):
        return self.best.exploit(states)

    @property
    def best(self):
        return self.sacs[self.scores.index(min(self.scores))]
        """
    #[0,1]を選択している部分？

    """
    #SACSを取り出す
    @property
    def pickup_sac(self):
        self.sorted_sacs = sorted(self.sacs, key=min(self.scores))
        return self.sorted_sacs
    """


    #簡単化して、多数のpathが指定されても問題がないmix
    def newmix(self, states):
        #逆数を取る際に、分母に加える定数
        dig1 = 1
        dig2 = 0.1
        dig3 = 0.01
        dig4 = 0.001
        dig5 = 0.0001
        dig=0.00000001
        dig = 10**(-15)

        #逆数をとる、総和を取る
        inver = [1/(dig + score) for score in self.scores]
        sum_wei = sum(inver)

        #重みを作る
        wei = [inv/sum_wei for inv in inver]

        exploits = [s.exploit(states) for s in self.sacs]

        np_wei = np.array(wei)
        np_exploits = np.array(exploits)

        for h, w in zip(self.w_history.values(), wei):
            h.append(w)


        return np.dot(np_wei, np_exploits)
    

    @property
    def e1(self):
        return self.sacs[self.scores.index(min(self.scores))]
    #予測誤差の小さい方のSAC
    
    @property
    def e2(self):
        return self.sacs[self.scores.index(max(self.scores))]
    #予測誤差が大きい方のSAC
    
    def mix(self, states):
        er1 = min(self.scores)
        er2 = max(self.scores)
        #予測誤差

        w1 = 1/(0.001+er1)
        w2 = 1/(0.001+er2)
        #逆数

        wei1 = w1/(w1 + w2)
        wei2 = w2/(w1 + w2)
        #重みの計算

        exploit1 = self.e1.exploit(states)
        exploit2 = self.e2.exploit(states)
        #予測誤差によるそれぞれの行動

        return wei1*exploit1 + wei2*exploit2
        #行動を合成



    def update(self, states, actions, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        inp = torch.cat((states, actions), dim = 0)
        outs = [sac.predictor(inp).cpu().detach().numpy() for sac in self.sacs]
        
        def mse(A,B):
            return ((A - B)**2).mean()
        
        def moving_ave(o, n):
            return self.alpha*o + (1-self.alpha)*n

        self.scores = [moving_ave(score, mse(out, next_states)) for score, out in zip(self.scores, outs)]
        for h, score in zip(self.history.values(), self.scores):
            h.append(score)

    @classmethod
    def load(cls, pathes, **kwargs):
        sacs = [MLPSAC.load(path) for path in pathes]
        return cls(sacs, names = pathes, **kwargs)
    


class MOESACS_TEST:
    """
    MLPの予測誤差を重み付けして行動を合成
    たぶんupdateはそのままでいい
    bestで選択しているから、そこをいじる？    
    
    """
    def __init__(self, sacs, alpha=0.8, names=None, device=torch.device('cpu')):
        self.sacs = sacs
        self.scores = [0 for _ in sacs]
        self.alpha = alpha
        self.device = device

        if not names:
            self.names = [str(i) for i in range(len(sacs))]
        self.names = names
    
        self.history = OrderedDict([(name,[]) for name in names])

    """def exploit(self, states):
        return self.best.exploit(states)

    @property
    def best(self):
        return self.sacs[self.scores.index(min(self.scores))]
        """
    #[0,1]を選択している部分？

    """
    #SACSを取り出す
    @property
    def pickup_sac(self):
        self.sorted_sacs = sorted(self.sacs, key=min(self.scores))
        return self.sorted_sacs
    """


    #簡単化して、多数のpathが指定されても問題がないmix
    """
    def newmix(self, states):
        #逆数を取る際に、分母に加える定数
        dig1 = 1
        dig2 = 0.1
        dig3 = 0.01
        dig4 = 0.001
        dig5 = 0.0001
        dig=0.00000001

        #逆数をとる、総和を取る
        inver = [1/(dig4 + score) for score in self.scores]
        sum_wei = sum(inver)

        #重みを作る
        wei = [inv/sum_wei for inv in inver]

        exploits = [s.exploit(states) for s in self.sacs]

        np_wei = np.array(wei)
        np_exploits = np.array(exploits)

        return np.dot(np_wei, np_exploits)
        """
    
    def mix_test(self, states, weight):
        exploits = [s.exploit(states) for s in self.sacs]
        
        np_wei = [weight, 1-weight]
        np_exploits = np.array(exploits)
        
        return np.dot(np_wei, np_exploits)




    def update(self, states, actions, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        inp = torch.cat((states, actions), dim = 0)
        outs = [sac.predictor(inp).cpu().detach().numpy() for sac in self.sacs]
        
        def mse(A,B):
            return ((A - B)**2).mean()
        
        def moving_ave(o, n):
            return self.alpha*o + (1-self.alpha)*n

        self.scores = [moving_ave(score, mse(out, next_states)) for score, out in zip(self.scores, outs)]
        for h, score in zip(self.history.values(), self.scores):
            h.append(score)

    @classmethod
    def load(cls, pathes, **kwargs):
        sacs = [MLPSAC.load(path) for path in pathes]
        return cls(sacs, names = pathes, **kwargs)
    




class MOESACS_SCORE:

    def __init__(self, sacs, alpha=0.8, names=None, device=torch.device('cpu')):
        self.sacs = sacs
        self.scores = [0 for _ in sacs]
        self.alpha = alpha
        self.device = device

        if not names:
            self.names = [str(i) for i in range(len(sacs))]
        self.names = names
    
        self.history = OrderedDict([(name,[]) for name in names])
        self.w_history = OrderedDict([(name,[]) for name in names])

    #簡単化して、多数のpathが指定されても問題がないmix
    def s_newmix(self, states, s_list):
        #逆数を取る際に、分母に加える定数
        dig1 = 1
        dig2 = 0.1
        dig3 = 0.01
        dig4 = 0.001
        dig5 = 0.0001
        dig=0.00000001

        #逆数をとる、総和を取る
        inver = [1/(dig4 + abs(score-s)) for score, s in zip(self.scores, s_list)]
        sum_wei = sum(inver)

        #重みを作る
        wei = [inv/sum_wei for inv in inver]

        exploits = [s.exploit(states) for s in self.sacs]

        np_wei = np.array(wei)
        np_exploits = np.array(exploits)

        for h, w in zip(self.w_history.values(), wei):
            h.append(w)


        return np.dot(np_wei, np_exploits)


    def update(self, states, actions, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        inp = torch.cat((states, actions), dim = 0)
        outs = [sac.predictor(inp).cpu().detach().numpy() for sac in self.sacs]
        
        def mse(A,B):
            return ((A - B)**2).mean()
        
        def moving_ave(o, n):
            return self.alpha*o + (1-self.alpha)*n

        self.scores = [moving_ave(score, mse(out, next_states)) for score, out in zip(self.scores, outs)]
        for h, score in zip(self.history.values(), self.scores):
            h.append(score)

    @classmethod
    def load(cls, pathes, **kwargs):
        sacs = [MLPSAC.load(path) for path in pathes]
        return cls(sacs, names = pathes, **kwargs)