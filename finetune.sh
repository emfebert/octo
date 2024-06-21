
#debug no wandb
export WANDB_MODE=disabled && python scripts/finetune.py \
  --config=scripts/configs/finetune_config.py:full,language_conditioned \
  --config.pretrained_path=hf://rail-berkeley/octo-small \
  --config.save_interval=10 \
  --config.eval_interval=5 \
  --config.save_dir=/home/app/data/octo_training_runs \
   --debug


