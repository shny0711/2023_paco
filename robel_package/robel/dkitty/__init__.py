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

"""Gym environment registration for DKitty environments."""

from robel.utils.registration import register
from robel.dkitty.marathon import FricInfo

#===============================================================================
# Stand tasks
#===============================================================================

# Default number of steps per episode.
_STAND_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DKittyStandFixed-v0',
    class_path='robel.dkitty.stand:DKittyStandFixed',
    max_episode_steps=_STAND_EPISODE_LEN)

register(
    env_id='DKittyStandRandom-v0',
    class_path='robel.dkitty.stand:DKittyStandRandom',
    max_episode_steps=_STAND_EPISODE_LEN)

register(
    env_id='DKittyStandRandomDynamics-v0',
    class_path='robel.dkitty.stand:DKittyStandRandomDynamics',
    max_episode_steps=_STAND_EPISODE_LEN)

#===============================================================================
# Orient tasks
#===============================================================================

# Default number of steps per episode.
_ORIENT_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DKittyOrientFixed-v0',
    class_path='robel.dkitty.orient:DKittyOrientFixed',
    max_episode_steps=_ORIENT_EPISODE_LEN)

register(
    env_id='DKittyOrientRandom-v0',
    class_path='robel.dkitty.orient:DKittyOrientRandom',
    max_episode_steps=_ORIENT_EPISODE_LEN)

register(
    env_id='DKittyOrientRandomDynamics-v0',
    class_path='robel.dkitty.orient:DKittyOrientRandomDynamics',
    max_episode_steps=_ORIENT_EPISODE_LEN)

#===============================================================================
# Walk tasks
#===============================================================================

# Default number of steps per episode.
_WALK_EPISODE_LEN = 160  # 160*40*2.5ms = 16s

register(
    env_id='DKittyWalkFixed-v0',
    class_path='robel.dkitty.walk:DKittyWalkFixed',
    max_episode_steps=_WALK_EPISODE_LEN)

register(
    env_id='DKittyWalkRandom-v0',
    class_path='robel.dkitty.walk:DKittyWalkRandom',
    max_episode_steps=_WALK_EPISODE_LEN)

register(
    env_id='DKittyWalkRandomDynamics-v0',
    class_path='robel.dkitty.walk:DKittyWalkRandomDynamics',
    max_episode_steps=_WALK_EPISODE_LEN)


for name, path in [
        ("Walk", 'robel.dkitty.walk:DKittyWalkFixed'),
        ("Marathon", "robel.dkitty.marathon:DKittyMarathonFixed"),
        ("Follow", "robel.dkitty.follow:DKittyFollowFixed"),
    ]:
    for fr in [0,250,500,1000,1500,2000,2500,5000]:
        register(
            env_id=f'{name}FR{fr}-v0',
            class_path=path,
            max_episode_steps=_WALK_EPISODE_LEN,
            kwargs={
                "falling_reward": -fr
            },
            )

    for fric in [1.0, 0.1, 0.01, 0.001]:
        register(
            env_id=f'{name}Fric{fric}-v0',
            class_path=path,
            max_episode_steps=_WALK_EPISODE_LEN,
            kwargs={
                "friction": fric,
            },
        )

register(
    env_id='MarathonU__H_ML-v1',
    class_path='robel.dkitty.marathon:MarathonLogSampleFriction',
    max_episode_steps=_WALK_EPISODE_LEN,
    kwargs={
        "high_fric": 0,
        "low_fric": -2,
    }
    )

register(
    env_id='MarathonC__H_L-v0',
    class_path='robel.dkitty.marathon:MarathonChangable',
    max_episode_steps=_WALK_EPISODE_LEN*2,
    kwargs={
        "frics":[
            FricInfo(-float("inf"), 3, 1.0),
            FricInfo(3, float("inf"), 0.001)
        ]
    }
    )

register(
    env_id='MarathonC__H_ML-v0',
    class_path='robel.dkitty.marathon:MarathonChangable',
    max_episode_steps=_WALK_EPISODE_LEN*2,
    kwargs={
        "frics":[
            FricInfo(-float("inf"), 2, 1.0),
            FricInfo(2, float("inf"), 0.01)
        ]
    }
    )