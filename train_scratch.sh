
#debug no wandb
export WANDB_MODE=disabled && python scripts/train.py \
 --config scripts/configs/act_pretrain_config.py:dummy \
  --config.save_interval=10 \
  --config.viz_interval=10 \
  --config.eval_interval=5 \
  --config.save_dir=/home/app/data/octo_training_runs \
   --debug

#debug with wandb
export WANDB_MODE=online && python scripts/train.py \
 --config scripts/configs/act_pretrain_config.py:dummy \
  --config.save_interval=10 \
  --config.viz_interval=10 \
  --config.eval_interval=5 \
  --config.save_dir=/home/app/data/octo_training_runs

python scripts/train.py \
 --config scripts/configs/act_pretrain_config.py:vit_s \
 --config.save_dir=/home/app/data/octo_training_runs

 