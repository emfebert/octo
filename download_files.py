import os


GRAB_ITERATIONS = [10000,
                   20000,
                   30000,
                   40000,
                   50000]

OTHER_FOLDERS_AND_FILES = [
    'config.json',
    'dataset_statistics.json',
    'example_batch.msgpack',
    'state'
]



def download_from_gcs(args):

    destination_path = args.destination_path + '/' + args.run_name
    source_path = args.source_path + '/' + args.run_name

    if not args.dry_run:
        os.makedirs(destination_path, exist_ok=True)
    else:
        print('would create', destination_path)

    for file in OTHER_FOLDERS_AND_FILES:
        # cmd = 'scp -r ' + args.machine + ':' + source_path + '/' + file + ' ' + destination_path
        cmd = 'gcloud compute scp --recurse ' + args.machine + ':' + source_path + '/' + file + ' ' \
              + destination_path + ' --zone=' + args.zone
        if not args.dry_run:
            os.system(cmd)
        else:
            print('would execute', cmd)

    for iteration in GRAB_ITERATIONS:
        # cmd = 'scp -r ' +  args.machine + ':' + source_path + '/' + iteration + ' ' + destination_path
        iteration = str(iteration)
        cmd = 'gcloud compute scp --recurse ' + args.machine + ':' + source_path + '/' + iteration + ' ' \
            + destination_path + ' --zone=' + args.zone
        if not args.dry_run:
            os.system(cmd)
        else:
            print('would execute', cmd)
    


if __name__ == "__main__":
    # example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', default='a100-3', type=str)
    parser.add_argument('--zone', default='us-central1-a', type=str)
    parser.add_argument('--run_name', type=str, required=True)   
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--destination_path', default='/home/user/data/octo_training_runs', type=str)
    parser.add_argument('--source_path', default='/home/febert/data/octo_training_runs/octo_finetune', type=str)

    args = parser.parse_args()

    download_from_gcs(args)