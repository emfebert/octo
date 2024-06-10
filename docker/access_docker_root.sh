docker run --gpus all -it --rm \
    --volume $CODE/octo:/home/app/octo \
    --volume $DATA/tensorflow_datasets:/home/app/data/tensorflow_datasets \
    --volume $DATA:/home/app/data \
    -e "DATA=/home/app/data/" \
    emancro/jax_mamba_root:latest /bin/bash -c "source activate octo && cd /home/app/octo && pip install -e . && /bin/bash"
