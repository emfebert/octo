import os


GRAB_ITERATIONS = [1500, 1000]

OTHER_FOLDERS_AND_FILES = [
    # -rw-r--r-- 1 root root   3592 Jun 10 20:54 config.json
    # -rw-r--r-- 1 root root   1795 Jun 10 20:54 dataset_statistics.json
    # -rw-r--r-- 1 root root 591976 Jun 10 20:54 example_batch.msgpack
    # drwxr-xr-x 3 root root   4096 Jun 10 21:10 state
    'config.json',
    'dataset_statistics.json',
    'example_batch.msgpack',
    'state'
]



def download_from_gcs(args):

    for file in OTHER_FOLDERS_AND_FILES:
        # cmd = f'rsync -av {args.machine}:/home/febert/data/octo_training_runs/octo/{args.run_name}/{file} /home/febert/data/octo_training_runs/octo/{args.run_name}'
        cmd = f'scp -r {args.machine}:/home/febert/data/octo_training_runs/octo/{args.run_name}/{file} /home/febert/data/octo_training_runs/octo/{args.run_name}'
        if not args.dry_run:
            os.system(cmd)
        else:
            print('would execute', cmd)

    for iteration in GRAB_ITERATIONS:
        # cmd = f'rsync -av {args.machine}:/home/febert/data/octo_training_runs/octo/{args.run_name}/{iteration} /home/febert/data/octo_training_runs/octo/{args.run_name}'
        cmd = f'scp -r {args.machine}:/home/febert/data/octo_training_runs/octo/{args.run_name}/{iteration} /home/febert/data/octo_training_runs/octo/{args.run_name}'
        if not args.dry_run:
            os.system(cmd)
        else:
            print('would execute', cmd)
    


if __name__ == "__main__":
    # example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', default='a100-1', type=str)
    parser.add_argument('--run_name', default='experiment_20240610_205136', type=str, required=True)   
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    download_from_gcs(args)