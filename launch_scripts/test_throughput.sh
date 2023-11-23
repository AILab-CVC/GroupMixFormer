ROOT=$1
CODE_BASE=$2
TAG=groupmixformer_small

cd $CODE_BASE && CUDA_VISIBLE_DEVICES=7 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
    --cfg configs/swin_tiny_patch4_window7_224.yaml \
    --data-path $ROOT/data/ILSVRC/Data/CLS-LOC \
    --batch-size 128 \
    --throughput \
    --disable_amp