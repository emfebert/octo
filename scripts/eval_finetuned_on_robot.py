"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng

import json

from octo.envs.easo_env import EasoGymEnvRel

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)


flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("window_size", 1, "Observation history length")
flags.DEFINE_integer(
    "action_horizon", 4, "Length of action sequence to execute/ensemble"
)


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


def main(_):

    env = EasoGymEnvRel()

    # load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )    

    # wrap the robot environment
    env = HistoryWrapper(env, FLAGS.window_size)
    # env = TemporalEnsembleWrapper(env, FLAGS.action_horizon)
    # switch TemporalEnsembleWrapper with RHCWrapper for receding horizon control
    # env = RHCWrapper(env, FLAGS.action_horizon)

    # create policy functions
    
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
            unnormalization_statistics=pretrained_model.dataset_statistics["action"],
        )
        # remove batch dim
        return actions[0]
    
    env.set_proprio_statistics(model.dataset_statistics["proprio"])

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            # argmax=FLAGS.deterministic,
            # temperature=FLAGS.temperature,
        )
    )
    
    

    # goal sampling loop
    while True:
        
        env.pre_position()
        
        text = 'blank'
        # Format task for the model
        task = model.create_tasks(texts=[text])

        # reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        # do rollout
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:

            # save images
            images.append(obs["image_wrist"][-1])

            if FLAGS.show_image:
                bgr_img = cv2.cvtColor(obs["image_wrist"][-1], cv2.COLOR_RGB2BGR)
                cv2.imshow("img_view", bgr_img)
                cv2.waitKey(20)

            # get action
            
            if t % FLAGS.action_horizon == 0:
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

            # perform environment step
            start_time = time.time()
            obs, _, _, truncated, _ = env.step(action[t % FLAGS.action_horizon])
            print("step time: ", time.time() - start_time)

            t += 1

            if truncated:
                break

        # save video
        # if FLAGS.video_save_path is not None:
        #     os.makedirs(FLAGS.video_save_path, exist_ok=True)
        #     curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     save_path = os.path.join(
        #         FLAGS.video_save_path,
        #         f"{curr_time}.mp4",
        #     )
        #     video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
        #     imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
