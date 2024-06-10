from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict

get_base_config = imp.load_source(
    "act_config", os.path.join(os.path.dirname(__file__), "act_config.py")
).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    config = get_base_config(config_string)

    config["window_size"] = 2 
    config["num_steps"] = 300000
    config["model"]["observation_tokenizers"] = {
        "wrist": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_wrist"],
            task_stack_keys=["image_wrist"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
    }
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizer,
            encoder="t5-base",
            finetune_encoder=False,
        ),
    }
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHead,
        readout_key="readout_action",
        use_map=False,
        pred_horizon=30,   # 4
        action_dim=10,
        dropout_rate=0.0,
    )

    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "wrist": (256, 256),
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = [
        wrist_augment_kwargs,
    ]

    config = update_config(
        config,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="act_full",
                data_dir=os.environ['DATA'] + "/tensorflow_datasets",
                load_camera_views=("wrist",),
                load_depth=False,
            ),
            traj_transform_kwargs=dict(
                future_action_window_size=31,
                task_augment_strategy="delete_and_rephrase",
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=0.5,
                ),
            ),
            frame_transform_kwargs=dict(
                image_dropout_prob=0.5,
            ),
            batch_size=64,  # 128
            shuffle_buffer_size=20000,  # 500000,
            balance_weights=True,
        ),
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(
            ModuleSpec.create(
                hf_weights_loader,
                hf_model="t5-base",
            ),
        ),
        eval_datasets=["insert_ibuprofen"],
    )

    return config
