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
    def __init__(self, main, sacs, ws=[1,0]):
        self.main = main
        self.sacs = sacs
        self.change_weight(self, ws)
    
    @staticmethod
    def load(cls, pathes):
        return cls(instant_algo(pathes[0]), SACS.load(pathes))
    
    def change_weight(self, ws):
        dic = {}
        for w, sac in zip(ws, self.sacs.sacs):
            for k,v in sac.state_dict.items():
                if k not in dic:
                    dic[k] = 0
                dic[k] += w*v
        self.main.load(dic)
    
    def update(self, state, action, next_state):
        self.sacs.update(state, action, next_state)
        ws = self.sacs.scores / (self.sacs.scores.sum() + 1e-6)
        self.change_weight(ws)

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