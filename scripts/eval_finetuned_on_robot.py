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
from octo.envs.easo_env import EasoGymEnv
import imageio
import jax
import jax.numpy as jnp
import numpy as np

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


flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("horizon", 2, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Length of action sequence from model")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")



def main(_):
    env = EasoGymEnv()
    
    # load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    # wrap the robot environment
    env = UnnormalizeActionProprio(
        env, model.dataset_statistics["insert_ibuprofen"], normalization_type="normal"
    )
    env = HistoryWrapper(env, FLAGS.horizon)
    # env = TemporalEnsembleWrapper(env, FLAGS.pred_horizon)
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
        
        # Format task for the model
        task = model.create_tasks(texts=['blank'])
        
        # input("Press [Enter] to start.")
        # env.pre_position()

        # reset env
        obs, _ = env.reset()
        
        
        
        # import matplotlib.pyplot as plt
        # print("obs image")
        # plt.imshow(obs['image_wrist'][0])
        # plt.show()
        
        
        import tensorflow as tf
        import flax
        # load example batch
        with tf.io.gfile.GFile(
            tf.io.gfile.join(FLAGS.checkpoint_weights_path, "example_batch.msgpack"), "rb"
        ) as f:
            example_batch = flax.serialization.msgpack_restore(f.read())
        # print('example batch', example_batch)
        
        example_observations = example_batch['observation']
        example_task = example_batch['task']
        
        
        # def print_infos(dict_):
        #     for key, value in dict_.items():
        #         print(key)
        #         if isinstance(value, dict):
        #             print_infos(dict_)
        #         else:
        #             print(f'{key}  shape: {value.shape}, min {value.min()} max {value.max()}')
        
        
        # print('env')
        # print_infos(obs)        
        # print_infos(task)
        
        # print('example')
        # print_infos(example_observations)        
        # print_infos(example_task)
        
        obs = example_batch['observation']
        task = example_batch['task']        
        obs = jax.tree_map(lambda x: x[0], obs)
        
        
        # print("example batch image")
        # import pdb; pdb.set_trace()
        # plt.imshow(example_batch['observation']['image_wrist'][0, 0])
        # plt.show()
        
        
        
        import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()
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
            
            if t % query_freq == 0:
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                
                print('action unnormalized: ', action[0])
                print("forward pass time: ", time.time() - forward_pass_time)
                import pdb; pdb.set_trace()

            # perform environment step
            start_time = time.time()
            obs, _, _, truncated, _ = env.step(action[t % query_freq])
            print("step time: ", time.time() - start_time)

            t += 1

            if truncated:
                break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / query_freq)


if __name__ == "__main__":
    app.run(main)
