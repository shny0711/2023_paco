ALL_STEPS = 10**6

hl_sac = SAC()

for i in range(ALL_STEPS):
    high_action = hl_sac.explore() # high_action は１次元で重みを表している
    n_obs, reward, done, info = high_env.step(high_action)




#
class HighEnv:
    def step(self, high_action):
        low_action = self.sacs.exploit(high_action) # low_actionは１２次元

        n_obs, reward, done, info = self.low_env.step(low_action)
        return n_obs, reward, done, info

class SACS:
    def exploit(self, high_action):
        sac1, sac2 = self.sacs
        low_action = high_action * sac1.exploit() + (1 - high_action) * sac2.exploit()
        return low_action