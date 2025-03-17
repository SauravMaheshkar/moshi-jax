from flax import nnx

from moshi_jax.configs.v0 import config_v0_1
from moshi_jax.modules.transformer import Transformer


def test_init():
    cfg = config_v0_1()
    module = Transformer(cfg, rngs=nnx.Rngs(default=42))

    assert module is not None
