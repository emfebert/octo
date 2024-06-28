import time

import gym
import numpy as np
from pyquaternion import Quaternion
import jax.numpy as jnp


from emancro_base.robot_infra.xarm.visio_motor.xarm_mdp_env import XarmMDP

from emancro_base.robot_infra.oculus_teleop.vr_teleop_policy import VRTeleopPolicy
from oculus_reader.reader import OculusReader
from skimage.transform import resize



# from scipy.spatial.transform import Rotation
# def euler_to_r6(euler, degrees=False):
#     rot_mat = Rotation.from_euler("xyz", euler, degrees=degrees).as_matrix()
#     a1, a2 = rot_mat[0], rot_mat[1]
#     return np.concatenate((a1, a2)).astype(np.float32)

# def get_qpos(xarm_robot_pose):
#     # goes together with the function below! Do not change separately!
#     assert xarm_robot_pose.shape == (6,)  # units : [mm, mm, mm, deg, deg, deg]
#     xyz = xarm_robot_pose[:3]
#     euler = xarm_robot_pose[3:]
#     r6s = euler_to_r6(euler, degrees=True)
#     return np.concatenate([xyz, r6s], axis=0)


def preproceess_obs(obs, size, proprio_statistics=None):
    curr_obs = {}
    old2new_imagenames = {"color_image": "image_wrist"}
    for old, new in old2new_imagenames.items():
        curr_image = obs.images[old]
        curr_obs[new] = resize(
            curr_image, (size, size), anti_aliasing=True
        )
        curr_obs[new] = (curr_obs[new]*255).astype(np.uint8)
        
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.imshow(curr_obs['image_wrist'])
    # plt.show()
        
    curr_obs["proprio"] = (obs.robot_pose - proprio_statistics["mean"])/ proprio_statistics["std"]
    # curr_obs["pad_mask_dict"] = {'image_wrist': jnp.array([ True]),
    #                              'language_instruction': jnp.array([True]),
    #                              'proprio': jnp.array([ True]),
    #                              'timestep': jnp.array([ True])}

    return curr_obs

class EasoGymEnvRel(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    Needed to use Gym wrappers.
    """

    def __init__(
        self,
        im_size: int = 128,
    ):
        self.im_size = im_size
        self.observation_space = gym.spaces.Dict(
            {
                "image_wrist": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((7,)) * -1, high=np.ones((7,)), dtype=np.float64
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64
        )
        
        self._env = XarmMDP(control_freq=50)    
        self.oculus_reader = OculusReader()
        self.teleop_policy = VRTeleopPolicy(environment=self._env, oculus_reader=self.oculus_reader)
        
        self.time_step = None
        
        
    def set_proprio_statistics(self, proprio_statistics):
        self.proprio_statistics = proprio_statistics
        print('proprio_statistics: ', proprio_statistics)
        

    def step(self, action):      
        self.time_step += 1  
        print("action to be taken: ", action)
        obs, info = self._env.step(action, self.time_step, action_mode='delta_pos_delta_rot')
        obs = preproceess_obs(obs, self.im_size, self.proprio_statistics)
        obs['timestep'] = self.time_step
        
        truncated = False
        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        self.time_step = 0
        super().reset(seed=seed)
        obs = self._env.reset()
        obs = preproceess_obs(obs, self.im_size, self.proprio_statistics)
        obs['timestep'] = self.time_step

        return obs, {}
    
    def pre_position(self):
        print('pre-positioning, press handle button to pre-position robot.Relase Handle Button to stop recording')
        self.teleop_policy.wait_for_start()
        print('started')
        
        done = False
        obs = self._env.reset()
        t = 0
        while not done:
            action_infos = self.teleop_policy.act(obs)            
            if action_infos['button_infos']['A']:
                done = True
            obs, info = self._env.step(action_infos, t, action_mode='delta_pos_abs_rot')
            
            t += 1
        print('pre-positioning done.')
