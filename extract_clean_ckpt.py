import torch
import os


if __name__ == '__main__':
    path = '/home/xieenze/ai-theory-enze-efs2/chongjian/output/GroupMixFormer/groupmixformer/groupmixformer_tiny/ckpt_epoch_298.pth'
    tgt_path = '/home/xieenze/ai-theory-enze-efs2/chongjian/output/GroupMixFormer/public'
    ckpt_name = 'groupmixformer_tiny.pth'

    tgt_ckpt_path = os.path.join(tgt_path, ckpt_name)

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = {'model': checkpoint['model']}

    torch.save(state_dict, tgt_ckpt_path)