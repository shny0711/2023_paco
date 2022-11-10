import robel
import gym
from mujoco_py.generated import const
import numpy as np

env = gym.make("MarathonC__H_L-v0")
for ep in range(10):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        # viewer = env.env.sim.render_contexts[0]
        # for v in env.env.sim.render_contexts:
        # v.add_marker(type=const.GEOM_BOX,
        #               pos=np.array([0, 1, .5]),
        #               label="hoge")
        act = env.action_space.sample()
        obs, reward, done, info = env.step(act)

        # del v._markers[:]
