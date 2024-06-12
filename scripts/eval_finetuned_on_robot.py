"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""
import dlimp as dl
from datetime import datetime
from functools import partial
import os
import time

import copy

from absl import app, flags, logging
import click
import cv2
from envs.widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import (
    HistoryWrapper,
    TemporalEnsembleWrapper,
    UnnormalizeActionProprio,
)

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)



flags.DEFINE_integer("im_size", 256, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("horizon", 1, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Length of action sequence from model")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
We also use a step duration of 0.4s to reduce the jerkiness of the policy.
Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.4
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################

def pre_position(env, teleop_policy):
    print('pre-positioning, press handle button to pre-position robot.Relase Handle Button to stop recording')
    teleop_policy.wait_for_start()
    print('started')
    
    done = False
    obs = env.reset()
    t = 0
    while not done:
        action_infos = teleop_policy.act(obs)            
        if action_infos['button_infos']['A']:
            done = True
        obs, info = env.step(action_infos, t)
        
        t += 1
    print('pre-positioning done.')


IMAGE_NAMES = ['image_wrist']

def preproceess_obs(obs, size):

    curr_obs = {}

    old2new_imagenames = {"color_image": "image_wrist"}
    for old, new in old2new_imagenames.items():
        curr_image = obs.images[old]
        curr_obs[new] = jnp.array(curr_image)
    curr_obs = dl.transforms.resize_images(
        curr_obs, match=curr_obs.keys(), size=(size)
    )
    return curr_obs


class UnnormalizeActionProprio():
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self,
        action_proprio_metadata: dict,
        normalization_type: str,
    ):
        self.action_proprio_metadata = jax.tree_map(
            lambda x: np.array(x),
            action_proprio_metadata,
            is_leaf=lambda x: isinstance(x, list),
        )
        self.normalization_type = normalization_type
        super().__init__()

    def unnormalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        if self.normalization_type == "normal":
            return np.where(
                mask,
                (data * metadata["std"]) + metadata["mean"],
                data,
            )
        elif self.normalization_type == "bounds":
            return np.where(
                mask,
                ((data + 1) / 2 * (metadata["max"] - metadata["min"] + 1e-8))
                + metadata["min"],
                data,
            )
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        if self.normalization_type == "normal":
            return np.where(
                mask,
                (data - metadata["mean"]) / (metadata["std"] + 1e-8),
                data,
            )
        elif self.normalization_type == "bounds":
            return np.where(
                mask,
                np.clip(
                    2
                    * (data - metadata["min"])
                    / (metadata["max"] - metadata["min"] + 1e-8)
                    - 1,
                    -1,
                    1,
                ),
                data,
            )
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    # def action(self, action):
    #     return self.unnormalize(action, self.action_proprio_metadata["action"])

    # def observation(self, obs):
    #     obs["proprio"] = self.normalize(
    #         obs["proprio"], self.action_proprio_metadata["proprio"]
    #     )
    #     return obs


def main(_):

    from emancro_base.robot_infra.xarm.visio_motor.xarm_mdp_env import XarmMDP_DeltaPosAbsRot
    env = XarmMDP_DeltaPosAbsRot(control_freq=50)    
    from emancro_base.robot_infra.oculus_teleop.vr_teleop_policy import VRTeleopPolicy
    from emancro_base.robot_infra.transform_publisher.transform_publisher.transform_broadcast import TransformPublisherNodeManager
    from oculus_reader.reader import OculusReader
    
    nodemanager = TransformPublisherNodeManager()
    oculus_reader = OculusReader()
    teleop_policy = VRTeleopPolicy(node_manager=nodemanager, environment=env, oculus_reader=oculus_reader)
    


    # load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    # wrap the robot environment
    normalizer = UnnormalizeActionProprio(model.dataset_statistics["insert_ibuprofen"], "normal")

    env = HistoryWrapper(env, FLAGS.horizon)
    env = TemporalEnsembleWrapper(env, FLAGS.pred_horizon)
    # switch TemporalEnsembleWrapper with RHCWrapper for receding horizon control
    # env = RHCWrapper(env, FLAGS.exec_horizon)

    # create policy function
    @jax.jit
    def sample_actions(
        pretrained_model: OctoModel,
        observations,
        tasks,
        rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
        )
        # remove batch dim
        return actions[0]

    def supply_rng(f, rng=jax.random.PRNGKey(0)):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            return f(*args, rng=key, **kwargs)

        return wrapped

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        )
    )

    query_freq = 30 # same as pred_horizon


    # goal sampling loop
    while True:

        text = "blank" # the same dummy instruction as used in training
        # Format task for the model
        task = model.create_tasks(texts=[text])

        input("Press [Enter] to start.")

        # reset env
        pre_position(env, teleop_policy)
        
        rollout_id += 0
        
        obs = env.reset()
        obs = copy.deepcopy(preproceess_obs(obs))
        time.sleep(2.0)

        # do rollout
        last_tstep = time.time()
        images = []
        
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()

                # save images
                images.append(obs.images.color_image)
                
                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs.images.color_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()

                if t % query_freq == 0:
                    action = normalizer.unnormalize(action, model.dataset_statistics["insert_ibuprofen"]["action"])
                    assert action.shape.shape[0] == query_freq

                obs = env.step(action[:, t % query_freq])
                obs = copy.deepcopy(preproceess_obs(obs))
                print("step time: ", time.time() - start_time)

                t += 1

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = images
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
