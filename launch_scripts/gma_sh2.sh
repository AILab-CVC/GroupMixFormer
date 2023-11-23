pkill -f scripts/cloud_start.py >/dev/null 2>&1
fuser -k /dev/nvidia* >/dev/null 2>&1
sleep 5s

CODE_PATH=/data/chongjian/code/GroupMixFormer/

cd $CODE_PATH && python multi_machine_start.py



