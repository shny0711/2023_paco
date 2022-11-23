"""Follow tasks with DKitty robots.

This is a single movement from an initial position to a target position.
"""

import abc
import collections
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from robel.components.tracking import TrackerComponentBuilder, TrackerState
from robel.dkitty.base_env import BaseDKittyUprightEnv, BaseDKittyFrictionEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path

from typing import List
from dataclasses import dataclass

DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_marathon-v0.xml'

DEFAULT_OBSERVATION_KEYS = (
    'root_pos',
    'root_euler',
    'kitty_qpos',
    'root_vel',
    'root_angular_vel',
    'kitty_qvel',
    'last_action',
    'upright',
    'heading',
)


class BaseDKittyFollow(BaseDKittyFrictionEnv, metaclass=abc.ABCMeta):
    """Shared logic for DKitty follow tasks."""

    def __init__(self,
                 asset_path: str = DKITTY_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 target_tracker_id: Optional[Union[str, int]] = None,
                 heading_tracker_id: Optional[Union[str, int]] = None,
                 frame_skip: int = 40,
                 upright_threshold: float = 0.9,
                 upright_reward: float = 1,
                 falling_reward: float = -500,
                 follow_scale: int = -10,
                 desired_vel: float = 1.0,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            target_tracker_id: The device index or serial of the tracking device
                for the target location.
            heading_tracker_id: The device index or serial of the tracking
                device for the heading direction. This defaults to the target
                tracker.
            frame_skip: The number of simulation steps per environment step.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
        """
        self._target_tracker_id = target_tracker_id
        self._heading_tracker_id = heading_tracker_id
        if self._heading_tracker_id is None:
            self._heading_tracker_id = self._target_tracker_id

        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            upright_threshold=upright_threshold,
            upright_reward=upright_reward,
            falling_reward=falling_reward,
            **kwargs)

        self._initial_target_pos = np.zeros(3)
        self._initial_heading_pos = None
        self.follow_scale = follow_scale
        self.desired_vel = np.array([0, desired_vel, 0])

    def _configure_tracker(self, builder: TrackerComponentBuilder):
        """Configures the tracker component."""
        super()._configure_tracker(builder)
        builder.add_tracker_group(
            'heading',
            hardware_tracker_id=self._heading_tracker_id,
            sim_params=dict(
                element_name='heading',
                element_type='site',
            ),
            mimic_xy_only=True)

    def _reset(self):
        """Resets the environment."""
        self._set_fric(self.base_friction)
        self._reset_dkitty_standing()

        # If no heading is provided, head towards the target.
        target_pos = self._initial_target_pos
        heading_pos = self._initial_heading_pos
        if heading_pos is None:
            heading_pos = target_pos

        # Set the tracker locations.
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot=np.identity(3)),
            'heading': TrackerState(pos=heading_pos),
        })

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        # Apply action.
        self.sim_scene.renderer.desired_vel = self.desired_vel
        self.robot.step({
            'dkitty': action,
        })

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        robot_state = self.robot.get_state('dkitty')
        heading_state, torso_track_state = self.tracker.get_state(
            ['heading', 'torso'])

        kitty_xy = torso_track_state.pos[:2]

        # Get the heading of the torso (the y-axis).
        current_heading = torso_track_state.rot[:2, 1]

        # Get the direction towards the heading location.
        desired_heading = heading_state.pos[:2] - kitty_xy

        # Calculate the alignment of the heading with the desired direction.
        heading = calculate_cosine(current_heading, desired_heading)

        return collections.OrderedDict((
            # Add observation terms relating to being upright.
            *self._get_upright_obs(torso_track_state).items(),
            ('root_pos', torso_track_state.pos),
            ('root_euler', torso_track_state.rot_euler),
            ('root_vel', torso_track_state.vel),
            ('root_angular_vel', torso_track_state.angular_vel),
            ('kitty_qpos', robot_state.qpos),
            ('kitty_qvel', robot_state.qvel),
            ('last_action', self._get_last_action()),
            ('heading', heading),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        follow_reward = np.linalg.norm(self.desired_vel[2] - obs_dict['root_vel'][0][2])

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            ("follow", self.follow_scale * follow_reward)
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('success', obs_dict['root_pos'][0][1] > 5.0),
        ))


@configurable(pickleable=True)
class DKittyFollowFixed(BaseDKittyFollow):
    """Follow straight towards a fixed location."""

    def _reset(self):
        """Resets the environment."""
        target_dist = 10.0
        target_theta = np.pi / 2  # Point towards y-axis
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()

@dataclass
class FricInfo:
    left: float
    right: float
    fric: float
    
    def is_in(self, y:float):
        return self.left <= y < self.right

class FollowChangable(DKittyFollowFixed):
    def __init__(self, frics: List[FricInfo], **kwargs):
        self.frics = frics
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        f = self.get_fric(info["obs/root_pos"][1])
        self._set_fric(f)
        info["changed_friction"] = f
        return obs, reward, done, info
    
    def get_fric(self, y):
        for fi in self.frics:
            if fi.is_in(y):
                return fi.fric
        print("warning: 対応する範囲がありません")
        return self.frics[-1].fric

@configurable(pickleable=True)
class DKittyFollowRandom(BaseDKittyFollow):
    """Follow straight towards a random location."""

    def _reset(self):
        raise Exception("not inplimented")
        super()._reset()


@configurable(pickleable=True)
class DKittyFollowRandomDynamics(DKittyFollowRandom):
    """Follow straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()
