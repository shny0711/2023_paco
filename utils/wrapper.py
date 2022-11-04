import gym
import numpy as np
import cv2
from typing import Iterable

class RecordEnv(gym.Wrapper):
    recording = False
    buffer = []
    infos = []

    def start(self, path:str) -> None:
        self.recording = True
        self.path = path
    
    def step(self, action):
        if self.recording:
            rgb = self.env.render(mode='rgb_array', camera_id = 1)
            self.buffer.append(rgb)
        return self.env.step(action)

    def stop(self) -> str:
        self.recording = False
        rbgs_to_video(self.path, self.buffer, infos = self.infos)
        self.buffer = []

class RecordInfoEnv(RecordEnv):
    info = []

    def step(self, action):
        self.infos.append(self.info)
        self.info = []

        return super().step(action)
    
    def write(self, info: Iterable[str]):
        self.info = info


BLACK = (0,0,0)
PADDING = 30
def rbgs_to_video(path, frames, infos = None):
    if infos:
        assert len(frames) == len(infos)
        note = np.ones((480,480,3)).astype(np.uint8) * 255
        frames = [np.concatenate((f,note),axis=1) for f in frames]
    
    fourcc = cv2.VideoWriter_fourcc(*"vp09")

    size = (960, 480) if infos else (480, 480)
    video = cv2.VideoWriter(path, fourcc, 50.0, size)

    for i,frame in enumerate(frames):
        im = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if infos:
            info = infos[i]
            for j,item in enumerate(info):
                cv2.putText(im, item, (480+10,j*20+25), cv2.FONT_HERSHEY_PLAIN, 1, BLACK, 1, cv2.LINE_AA)

        video.write(im)
    video.release()