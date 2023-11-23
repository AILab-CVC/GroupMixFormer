import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from einops import rearrange
from functools import partial
from torch import nn, einsum

__all__ = [
    "groupmixformer_tiny",
    "groupmixformer_small",
    "groupmixformer_base",
]


class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Agg_0(nn.Module):
    def __init__(self, seg_dim):
        super().__init__()
        self.conv = SeparableConv2d(seg_dim * 3, seg_dim, 3, 1, 1)
        self.norm = nn.LayerNorm(seg_dim)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = self.act(self.norm(x.reshape(b, c, -1).permute(0, 2, 1)))

        return x


class Aggregator(nn.Module):
    def __init__(self, dim, seg=4):
        super().__init__()
        self.dim = dim
        self.seg = seg

        seg_dim = self.dim // self.seg

        self.norm0 = nn.SyncBatchNorm(seg_dim)
        self.act0 = nn.Hardswish()

        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.SyncBatchNorm(seg_dim)
        self.act1 = nn.Hardswish()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.SyncBatchNorm(seg_dim)
        self.act2 = nn.Hardswish()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        self.norm3 = nn.SyncBatchNorm(seg_dim)
        self.act3 = nn.Hardswish()

        self.agg0 = Agg_0(seg_dim)


    def forward(self, x, size, num_head):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        seg_dim = self.dim // self.seg

        x = x.split([seg_dim]*self.seg, dim=1)

        x_local = x[4].reshape(3, B//3, seg_dim, H, W).permute(1,0,2,3,4).reshape(B//3, 3*seg_dim, H, W)
        x_local = self.agg0(x_local)

        x0 = self.act0(self.norm0(x[0]))
        x1 = self.act1(self.norm1(self.agg1(x[1])))
        x2 = self.act2(self.norm2(self.agg2(x[2])))
        x3 = self.act3(self.norm3(self.agg3(x[3])))

        x = torch.cat([x0, x1, x2, x3], dim=1)

        C = C // 5 * 4
        x = x.reshape(3, B//3, num_head, C//num_head, H*W).permute(0, 1, 2, 4, 3)

        return x, x_local


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        super().__init__()

        if isinstance(window, int):
            window = {window: h}  # Set the same window size for all attention heads.
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch,
                                 kernel_size=(cur_window, cur_window),
                                 padding=(padding_size, padding_size),
                                 dilation=(dilation, dilation),
                                 groups=cur_head_split * Ch,
                                 )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)

        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):

        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W

        # Convolutional relative position encoding.
        q_img = q  # Shape: [B, h, H*W, Ch].
        v_img = v  # Shape: [B, h, H*W, Ch].

        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)  # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)  # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q_img * conv_v_img

        return EV_hat_img


class EfficientAtt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregator = Aggregator(dim=dim, seg=5)

        trans_dim = dim // 5 * 4
        self.crpe = ConvRelPosEnc(Ch=trans_dim // num_heads, h=num_heads, window={3: 2, 5: 3, 7: 3})

    def forward(self, x, size):
        B, N, C = x.shape

        # Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).reshape(3*B, N, C)

        qkv, x_agg0 = self.aggregator(qkv, size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # att
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        eff_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        # Merge and reshape.
        x = self.scale * eff_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C//5*4)
        x = torch.cat([x, x_agg0], dim=-1)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        # Depthwise convolution.
        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvStem(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, in_dim=3, embedding_dims=64):
        super().__init__()
        mid_dim = embedding_dims // 2

        self.proj1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.SyncBatchNorm(mid_dim)
        self.act1 = nn.Hardswish()

        self.proj2 = nn.Conv2d(mid_dim, embedding_dims, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.SyncBatchNorm(embedding_dims)
        self.act2 = nn.Hardswish()

    def forward(self, x):
        x = self.act1(self.norm1(self.proj1(x)))
        x = self.act2(self.norm2(self.proj2(x)))
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class PatchEmbedLayer(nn.Module):
    def __init__(self, patch_size=16, in_dim=3, embedding_dims=768, is_first_layer=False):
        super().__init__()
        if is_first_layer:
            patch_size = 1
            in_dim = embedding_dims

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = SeparableConv2d(in_dim, embedding_dims, 3, patch_size, 1)
        self.norm = nn.SyncBatchNorm(embedding_dims)
        self.act = nn.Hardswish()

    def forward(self, x):
        _, _, H, W = x.shape
        out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]
        x = self.act(self.norm(self.proj(x)))
        x = x.flatten(2).transpose(1, 2)
        return x, (out_H, out_W)


class GMA_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.cpe = ConvPosEnc(dim=dim, k=3)
        self.norm1 = norm_layer(dim)
        self.att = EfficientAtt(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,)
        self.drop_path_rate = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_input, size):
        x = self.cpe(x_input, size)
        cur = self.norm1(x)
        cur = self.att(cur, size)
        x = x + self.drop_path_rate(cur)

        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path_rate(cur)

        return x


class GMA_Stage(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 serial_depth=None):
        super().__init__()

        self.serial_depth = serial_depth
        self.gma_stage = nn.ModuleList([
            GMA_Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path_rate=drop_path_rate[i], act_layer=act_layer, norm_layer=norm_layer,
            ) for i in range(serial_depth)]
        )

    def forward(self, x, size):
        for i in range(self.serial_depth):
            x = self.gma_stage[i](x, size)
        return x


def stochastic_depth(drop_path_rate, serial_depths, num_stages):
    dpr  = [x.item() for x in torch.linspace(0, drop_path_rate, sum(serial_depths))]
    index_list = [0] + serial_depths
    dpr_per_stage = [dpr[sum(index_list[:i]): sum(index_list[:i+1])] for i in range(1,num_stages+1)]

    return dpr_per_stage


class GroupMixFormer(nn.Module):
    def __init__(
            self,
            patch_size=4, # seem useless
            in_dim=3, # seem useless
            num_stages = 4,
            num_classes=1000,
            embedding_dims= [80, 160, 320, 320],
            serial_depths=[2, 4, 12, 4],
            num_heads=8,
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            return_interm_layers=False,
            out_features=None, # seem useless
            pretrained=False  # seem useless
    ):
        super().__init__()

        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.conv_stem = ConvStem(in_dim=in_dim, embedding_dims=embedding_dims[0])

        self.patch_embed_layers = nn.ModuleList([
            PatchEmbedLayer(
                patch_size=2,
                in_dim=embedding_dims[i-1],
                embedding_dims=embedding_dims[i],
                is_first_layer=True if i == 0 else False,
            ) for i in range(self.num_stages)
        ])

        # Enable stochastic depth.
        dpr_per_stage = stochastic_depth(drop_path_rate, serial_depths, num_stages)

        # Serial blocks 1.
        self.groupmixformer_backbone = nn.ModuleList([
            GMA_Stage(
                dim=embedding_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr_per_stage[i],
                norm_layer=norm_layer,
                serial_depth=serial_depths[i]
            ) for i in range(self.num_stages)
        ])

        # Classification head(s).
        if not self.return_interm_layers:
            self.norm4 = nn.SyncBatchNorm(embedding_dims[3])
            self.head = nn.Linear(embedding_dims[3], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embedding_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        b, _, _, _ = x.shape
        x = self.conv_stem(x)
        out = []

        for i in range(self.num_stages):
            x_patch, (H, W) = self.patch_embed_layers[i](x)
            x = self.groupmixformer_backbone[i](x_patch, (H, W))
            x = x.reshape(b, H, W, -1).permute(0, 3, 1, 2)
            out.append(x)

        return out

    def forward(self, x):
        if self.return_interm_layers:  # Return intermediate features (for down-stream tasks).
            return self.forward_features(x)
        else:  # Return features for classification.
            x = self.forward_features(x)

            x = self.norm4(x[-1])
            x = x.mean(dim=(2, 3))
            x = self.head(x)
            return x


if __name__ == "__main__":
    device = 'cuda'
    model = GroupMixFormer().to(device)
    model.eval()
    inputs = torch.randn(1, 3, 224, 224).to(device)
    model(inputs)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

    print(flop_count_table(flops, max_depth=1))

