from .groupmixformer import GroupMixFormer

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'groupmixformer':
        model = GroupMixFormer(
            patch_size=config.MODEL.GMA.PATCH_SIZE,
            in_dim=config.MODEL.GMA.IN_DIM,
            num_stages=config.MODEL.GMA.NUM_STAGES,
            num_classes=config.MODEL.GMA.NUM_CLASSES,
            embedding_dims=config.MODEL.GMA.EMBEDDING_DIMS,
            serial_depths=config.MODEL.GMA.SERIAL_DEPTHS,
            num_heads=config.MODEL.GMA.NUM_HEADS,
            mlp_ratios=config.MODEL.GMA.MLP_RATIOS,
            qkv_bias=config.MODEL.GMA.QKV_BIAS,
            qk_scale=config.MODEL.GMA.QKV_SCALE,
            drop_rate=config.MODEL.GMA.DROP_RATE,
            attn_drop_rate=config.MODEL.GMA.ATTN_DROP_RATE,
            drop_path_rate=config.MODEL.GMA.DROP_PATH_RATE,
            pretrained=config.MODEL.GMA.PRETRAINED
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
