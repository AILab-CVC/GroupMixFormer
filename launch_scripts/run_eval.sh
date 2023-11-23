ROOT=$1
CODE_BASE=$2
TAG=groupmixformer_tiny


cd $CODE_BASE && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --use_env test.py \
  --data-path $ROOT/data/ILSVRC/Data/CLS-LOC \
  --batch-size 64 \
  --output $ROOT/chongjian/output/GroupMixFormer \
  --cfg ./configs/$TAG.yaml \
  --model-type groupmixformer \
  --model-file groupmixformer.py \
  --tag $TAG
