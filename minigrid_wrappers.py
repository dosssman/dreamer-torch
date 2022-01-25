import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np
from gym import spaces

import cv2; cv2.ocl.setUseOpenCL(False)
    
class RGBImgFullGridWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use the full grid as seen by a human observer as input for the agent
    """
    def __init__(self, env, image_size=(84,84)):
        super().__init__(env)
        self.image_size = image_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(*self.image_size, 3),
            dtype='uint8'
        )
    
    def observation(self, obs):
        env = self.unwrapped

        rgb_full_obs = env.render(mode="rgb_array", highlight=False)
        rgb_full_obs = np.array(
            cv2.resize(rgb_full_obs,
                self.image_size,
                interpolation=cv2.INTER_AREA)
            )
        
        return {
            'mission': obs['mission'],
            'image': rgb_full_obs
        }

class RGBImgResizeWrapper(gym.core.ObservationWrapper):
    """
    To be called after RGBImgPartialObsWrapper to change the dimension
    of the pixel-based observation.
    Allows to keep the same CNN structure across tasks
    """
    def __init__(self, env, image_size=(84,84)):
        super().__init__(env)
        self.image_size = image_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(*self.image_size, 3),
            dtype='uint8'
        )
    
    def observation(self, obs):
        resized_img_obs = np.array(
            cv2.resize(obs['image'],
                self.image_size,
                interpolation=cv2.INTER_AREA)
            )
        
        return {
            'mission': obs['mission'],
            'image': resized_img_obs
        }

class ChannelFirstImgWrapper(gym.core.ObservationWrapper):
    """
    Wrapper that swaps the input dimension of [H,W,C] into [C,H,W]
    for Pytorch CNN compatibility
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.__class__ in [ImgObsWrapper], \
            f"Invalid environment class passed to ChannelFirstImgWrapper: {env.__class__}"
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype='uint8'
        )
    
    def observation(self, obs):
        return obs.transpose(2,0,1)

# Dictionary that maps following
# https://github.com/maximecb/gym-minigrid/blob/6f5fe8588d05eb13a08f971fd3c7a82c404dc1bb/gym_minigrid/minigrid.py#L628
# left: 0
# right: 1
# forward: 2
# pickup: 3
# drop: 4
# toggle: 5
# done: 6 (unused in most of the case, according to)
ACTIONS_DICT = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6
}
class ActionMaskingWrapper(gym.core.Wrapper):
    def __init__(self, env, invalid_actions_list=["done"]):
        super().__init__(env)
        self.valid_actions = {}
        self.valid_actions_real_idx = {}
        i = 0
        for k,v in ACTIONS_DICT.items():
            if k not in invalid_actions_list:
                self.valid_actions[k] = v
                self.valid_actions_real_idx[i] = v
                i += 1
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
    
    def _action_translation(self, action):
        """
            Will match the action passed to this wrapper to the underlying action space
            that is defined for most, if not all the minigrid envs.
        """
        return self.valid_actions_real_idx[action]

    def step(self, action):
        observation, reward, done, info = self.env.step(self._action_translation(action))
        return observation, reward, done, info

class FactoredStateRepWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use the customized, factored state representation for Four Room mainly
    """
    def __init__(self, env):
        super().__init__(env)

        factord_state_shape = self.unwrapped.env_data["factored_state_shape"]
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(factord_state_shape,),
            dtype='uint8'
        )
    
    def observation(self, obs):
        env = self.unwrapped

        factored_state, hl_factored_state = env.gen_factored_state_representation()

        return factored_state

class RenderWithoutHighlightWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def render(self, mode="rgb_array"):
        env = self.unwrapped
        return env.render(mode, highlight=False)

class DictionaryObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        return {"image": obs}  
        