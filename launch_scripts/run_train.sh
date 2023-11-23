ROOT=$1
CODE_BASE=$2
TAG=groupmixformer_small

cd $CODE_BASE &&  python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --use_env train.py \
  --data-path $ROOT/data/ILSVRC/Data/CLS-LOC \
  --batch-size 64 \
  --output $ROOT/chongjian/output/GroupMixFormer \
  --cfg ./configs/groupmixformer_small.yaml \
  --model-type groupmixformer \
  --model-file groupmixformer.py \
  --tag $TAG
