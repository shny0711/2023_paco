import gym
import numpy as np
import cv2
from typing import Iterable, List
from gym.core import Env
from dataclasses import dataclass

class HighRecordEnv(gym.Wrapper):
    def __init__(self, env: Env, hl_sac, camera_id=None) -> None:
        self.recording = False
        self.buffer = []
        self.infos = []
        self.hl_sac = hl_sac
        self.camera_id = camera_id
        super().__init__(env)

    def start(self, path:str) -> None:
        self.recording = True
        self.path = path

    # def exploits(self, state, sacs, high_action):
    #     low_actions = [sac.exploit(state) for sac in sacs]
    #     low_action = high_action * low_actions[0] +(1-high_action)*low_actions[1]
    #     return low_action
    
    def step(self, state, sacs, high_action):
        if self.recording:
            if self.camera_id:
                rgb = self.env.render(mode='rgb_array', camera_id=self.camera_id)
            else:
                rgb = self.env.render(mode='rgb_array')
            self.buffer.append(rgb)
        return self.env.step(state, sacs, high_action)

    def stop(self):
        self.recording = False
        rbgs_to_video(self.path, self.buffer, infos = self.infos)
        self.buffer = []


class HighRecordInfoEnv(HighRecordEnv):
    def __init__(self, env: Env, hl_sac,**kwargs) -> None:
        self.info = []
        self.hl_sac = hl_sac
        super().__init__(env, hl_sac, **kwargs)

    def step(self, state, sacs, high_action):

        self.infos.append(self.info)
        self.info = []

        return super().step(state, sacs, high_action)
    
    def write(self, info: Iterable[str]):
        self.info = info


BLACK = (0,0,0)
PADDING = 30
def rbgs_to_video(path, frames, infos = None):
    basesize = frames[0].shape[:2]

    if infos:
        assert len(frames) == len(infos)
        note = np.ones((*basesize,3)).astype(np.uint8) * 255
        frames = [np.concatenate((f,note),axis=1) for f in frames]
    
    fourcc = cv2.VideoWriter_fourcc(*"vp09")

    size = (basesize[1]*2, basesize[0]) if infos else basesize
    video = cv2.VideoWriter(path, fourcc, 50.0, size)

    for i,frame in enumerate(frames):
        im = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if infos:
            info = infos[i]
            for j,item in enumerate(info):
                cv2.putText(im, item, (480+10,j*20+25), cv2.FONT_HERSHEY_PLAIN, 1, BLACK, 1, cv2.LINE_AA)

        video.write(im)
    video.release()

class HighInfoEnv(gym.Wrapper):
    def reset(self):
        self.ret = 0
        self.t = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.ret += reward
        self.t += 1
        info["return"] = self.ret        
        info["step"] = self.t
        info["truncated"] = done and self.t >= self.env._max_episode_steps
        return obs, reward, done, info