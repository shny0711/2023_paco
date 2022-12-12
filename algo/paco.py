from algo.sac import SAC, MLPSAC, SACS
import torch
import numpy as np

cls_dict = {
    "MLPSAC": {
        "cls": MLPSAC,
        "path": [
            "actor.pth",
            "critirc.pth",
            "critic_target.pth",
            "predictor.pth"
        ]
    }
}

class MixSAC:
    module_names = ["critic", "critic_target", "actor"]
    def __init__(self, main, sacs, ws=[1,0]):
        self.main = main
        self.sacs = sacs
        self.change_weight(ws)
    
    @classmethod
    def load(cls, pathes, sacs_kwarg = {}, **kwargs):
        return cls(instant_algo(pathes[0]), SACS.load(pathes, **sacs_kwarg), **kwargs)
    
    def get_params(self):
        try:
            return self.params
        except:
            self.params = {m: [getattr(sac, m).state_dict() for sac in self.sacs.sacs] for m in self.module_names}
            return self.params
    
    def change_weight(self, ws, target_module = ["actor"]):
        self.ws = ws
        for module_name, state_dicts in self.get_params().items():
            if module_name not in target_module:
                continue
            new_state_dict = {}
            for w, state_dict in zip(ws, state_dicts):
                for k,v in state_dict.items():
                    if k not in new_state_dict:
                        new_state_dict[k] = 0
                    new_state_dict[k] += w*v
            getattr(self.main, module_name).load_state_dict(new_state_dict)
    
    def update(self, state, action, next_state):
        self.sacs.update(state, action, next_state)
        ws = self.sacs.scores / (sum(self.sacs.scores) + 1e-6)
        self.change_weight(ws)

class PACO:
    class BASE(MLPSAC):
        """ 基底となる部分で実際にSACのパラメータを持っている

        SACのAPIのうちパラメータをアップデートする箇所が必要
        """
        def update(self, states, actions, rewards, dones, next_states):
            """ 各タスク固有のリプレイバッファから取ったデータを使って学習
            """
            self.update_critic(states, actions, rewards, dones, next_states)
            self.update_actor(states)
            self.update_target()
            self.update_predictor(states, actions, next_states)
    
    class AGENT:
        """ BASEの線形和でタスク特化の専門エージェントになる

        SACのAPIのうち環境とインタラクトする箇所が必要
        """
        def __init__(self, base_num):
            self.ws = torch.nn.Linear(1, base_num, bias=False)
            self.optimizer = torch.optim.Adam(self.ws)
        
        def explore(self):
            pass

        def update_weight(self, states, actions, rewards, dones, next_states):
            pass

    def __init__(self, K, obs_shape, act_shape, T):
        """
        K 基底の数
        T タスクの数
        """
        
        self.base_agents = [PACO.BASE(obs_shape, act_shape) for k in range(K)]

    def update_base(self, loss):
        pass

cls_dict = {
    "MLPSAC": MLPSAC,
    "SAC": SAC
}
def instant_algo(path):
    for k,v in cls_dict.items():
        if k in path:
            return v.load(path)
    else:
        raise Exception(cls_dict, path)

if __name__ == "__main__":
    pathes = [
        "data/MarathonFric0.01-v0/MLPSAC/20221114_113841",
        "data/MarathonFric1.0-v0/MLPSAC/20221116_123558/best"
    ]
    mixsac = MixSAC(pathes)