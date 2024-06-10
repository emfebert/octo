
python scripts/train.py --config scripts/configs/act_pretrain_config.py:dummy

python scripts/train.py --config scripts/configs/act_pretrain_config.py:vit_s --config.save_dir=/home/app/data/octo_training_runs


python scripts/train.py --config scripts/configs/act_pretrain_config.py:vit_s --config.save_dir=/home/app/data/octo_training_runs --config.save_interval=10 --config.viz_interval=10 --debug