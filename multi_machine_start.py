import os,sys,shutil
import subprocess
import os
import argparse
import random
from cloud_tools.cloud.utils.dist_utils import synchronize_ip

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--np', type=int, default=8, help="nproc_per_node, number of GPUs per node")
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--main_port', type=str, default='20420', help='Port of the current rank 0.')
    parser.add_argument('--machine_num', type=int, default=2, help='number of machine')
    parser.add_argument('--vis_gpu', type=str, default='0,1,2,3,4,5,6,7', help='visible gpu id')
    parser.add_argument('--bucket', type=str, default='cneast3')

    # exps related
    parser.add_argument('--tag', type=str, default='debug')
    parser.add_argument('--code_path', type=str, default='/data/chongjian/code/GroupMixFormer/')
    parser.add_argument('--output', type=str, default='/data/chongjian/output/GroupMixFormer')
    parser.add_argument('--data_path', type=str, default='/data/data/ILSVRC/Data/CLS-LOC')
    parser.add_argument('--config', type=str, default='./configs/groupmixformer_small.yaml')
    parser.add_argument('--batchsize', type=int, default=128, help='number of batchsize')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    gpu_num = args.np
    WORK_DIR = os.getcwd()
    print("currrent working dir: ", WORK_DIR)
    sys.path.insert(0, os.getcwd())

    bucket = f"bucket-{args.bucket}"
    user_name = 'chongjiange'
    s3_work_dir = f's3://{bucket}/{user_name}/code/console/ip_temp'
    machine_num = args.machine_num
    master_addr, node_rank, host_ip = synchronize_ip(s3_work_dir, machine_num)
    master_addr, master_port = master_addr.split(':')
    print(f'master_addr:{master_addr}; master_port:{master_port}; node_rank:{node_rank} ')

    run_cmd = f'cd {args.code_path} && OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node {args.np}  ' \
              f'--nnodes={machine_num} --master_port {master_port} --node_rank={node_rank} --master_addr={master_addr} --use_env main.py ' \
              f'--data-path {args.data_path} ' \
              f'--batch-size {args.batchsize} ' \
              f'--output {args.output} ' \
              f'--cfg {args.config} ' \
              f'--model-type groupmixformer ' \
              f'--model-file groupmixformer.py ' \
              f'--tag {args.tag}'

    print(run_cmd)
    subprocess.call(run_cmd, shell=True)