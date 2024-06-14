docker run --gpus all -it --rm \
    --volume $CODE/octo:/home/app/octo \
    --volume /home/user/ros2_ws/src/base_repo3:/home/app/base_repo3 \
    --volume /home/user/software/xarm/software_upgrade/xarmstudio-x86_64-2.0.3/linux/xarm/software/xArm-Python-SDK:/home/app/xArm-Python-SDK \
    --volume $DATA/tensorflow_datasets:/home/app/data/tensorflow_datasets \
    --volume $DATA:/home/app/data \
    --volume /home/user/ros2_ws/src/oculus_reader/:/home/app/oculus_reader \
    -e "DATA=/home/app/data" \
    --volume /dev:/dev \
    -e PYTHONPATH=/home/app/base_repo3 \
    --privileged \
    jax_mamba_eval:latest /bin/bash -c "source activate octo && \
                                         cd /home/app/octo && \
                                          pip install -e . \
                                          cd /home/app/xArm-Python-SDK && \
                                          pip install -e . \
                                          cd /home/app/oculus_reader \
                                          pip install -e . \
                                          && /bin/bash"


