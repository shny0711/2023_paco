# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rendering simulation using mujoco_py."""

import mujoco_py
import numpy as np
from mujoco_py.generated import const
from robel.simulation.renderer import Renderer, RenderMode
from transforms3d.euler import euler2mat

class MjPyRenderer(Renderer):
    """Class for rendering mujoco_py simulations."""

    def __init__(self, sim):
        assert isinstance(sim, mujoco_py.MjSim), \
            'MjPyRenderer takes a mujoco_py MjSim object.'
        super().__init__(sim)
        self._onscreen_renderer = None
        self._offscreen_renderer = None

    def render_to_window(self):
        """Renders the simulation to a window."""
        if not self._onscreen_renderer:
            self._onscreen_renderer = mujoco_py.MjViewer(self._sim)
            self._update_camera_properties(self._onscreen_renderer.cam)

        self._onscreen_renderer.render()

    def refresh_window(self):
        """Refreshes the rendered window if one is present."""
        if self._onscreen_renderer is None:
            return
          
        self.set_marker(self._onscreen_renderer)
        self._onscreen_renderer.render()
        del self._onscreen_renderer._markers[:]

    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a numpy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A numpy array of the pixels.
        """
        assert width > 0 and height > 0
        if not self._offscreen_renderer:
            self._offscreen_renderer = mujoco_py \
                .MjRenderContextOffscreen(self._sim, device_id=-1)

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera_properties(self._offscreen_renderer.cam)

        self.set_marker(self._offscreen_renderer)
        self._offscreen_renderer.render(width, height, camera_id)
        del self._offscreen_renderer._markers[:]

        if mode == RenderMode.RGB:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=False)
            # Original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == RenderMode.DEPTH:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=True)[1]
            # Original image is upside-down, so flip it
            return data[::-1, :]
        else:
            raise NotImplementedError(mode)
    
    def set_marker(self, viewer):
        for i in range(10):
            viewer.add_marker(type=105,
                            pos=[0, i, .2],
                            label=f"{i}m")
        
        pos = self._sim.data.qpos[:3]
        def add_arrow(vel, height, color, label):
            x,y,z = vel
            size = np.array((0.02, 0.02, np.linalg.norm(vel)*1.0))
            viewer.add_marker(type=const.GEOM_ARROW,
                    pos=[pos[0], pos[1], height],
                    label=label,
                    mat=euler2mat(np.arctan2(y,x)+np.pi, np.arctan2(y,np.sqrt(x**2+y**2)), 0),
                    size=size,
                    rgba=color,
                    emission=1)
        
        add_arrow(self._sim.data.qvel[:3], 0.5, (0, 1, 0, 0.8), "torso velocity")
        if hasattr(self, "desired_vel"):
            add_arrow(getattr(self, "desired_vel"), 0.75, (1, 0, 0, 0.8), "desired velocity")