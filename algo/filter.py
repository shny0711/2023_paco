import numpy as np

class IdentityFilter:
    def reset(self):
        return None

    def apply(self, act):
        return act

class MovingFilter:
    def __init__(self, num, act_dim = 12):
        self.num = num
        self.act_dim = 12
        self.lis = np.zeros((num, act_dim))
        self.i = 0
    
    def reset(self):
        self.lis = np.zeros((self.num, self.act_dim))
        self.i = 0

    
    def apply(self, act):
        self.lis[self.i % self.num] = act
        self.i += 1

        # if self.i < self.num:
        #     return self.lis[:self.i].mean()

        return self.lis[:min(self.i,self.num)].mean(axis=0)

class WeightedMovingFilter(MovingFilter):
    def __init__(self, weight, **kwargs):
        super().__init__(len(weight), **kwargs)
        self.weight = weight
        print("filter sum", weight.sum())
    
    def apply(self, act):
        """ 最初ちょっと出力が弱まるな
        """
        self.lis = np.roll(self.lis, 1)
        self.lis[0] = act

        return np.sum(self.lis * self.weight.reshape(-1,1), axis=0)

if __name__ == "__main__":
    filter = WeightedMovingFilter(np.array([0.5, 0.25, 0.125, 0.0625, 0.0625]))
    for i in range(10):
        act = np.random.rand(12)
        filter.apply(act)