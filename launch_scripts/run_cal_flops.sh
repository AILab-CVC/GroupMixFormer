ROOT=$1
CODE_BASE=$2
TAG=groupmixformer_large

cd $CODE_BASE && CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --master_port 12347 \
  cal_flops.py  \
  --data-path $ROOT/data/ILSVRC/Data/CLS-LOC \
  --batch-size 2 \
  --output $ROOT/chongjian/output/GroupMixFormer \
  --cfg ./configs/$TAG.yaml \
  --model-type groupmixformer \
  --model-file groupmixformer.py \
  --tag $TAG