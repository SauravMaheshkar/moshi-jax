import ml_collections


def config_v0_1() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.dim = 4096
    config.num_heads = 32
    config.num_layers = 12
    config.mlp_dim = 4096 * 4
    config.kv_repeat = 1
    config.bias_attn = False
    config.bias_mlp = False
    config.layer_scale = None
    config.context = 3000
    config.gating = True
    config.norm = "rms_norm"

    return config
