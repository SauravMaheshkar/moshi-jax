from typing import Optional

import jax
import jax.numpy as jnp
import ml_collections
from flax import nnx


class Identity(nnx.Module):
    def __call__(self, xs: jax.Array) -> jax.Array:
        return xs


class LayerScale(nnx.Module):
    def __init__(self, dim: int):
        self.scale = jnp.ones((dim,))

    def __call__(self, xs: jax.Array) -> jax.Array:
        return xs * self.scale


class Attention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        kv_repeat: int,
        dim: int,
        context: int,
        bias: bool,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.kv_repeat = kv_repeat
        self.context = context
        self.dim = dim
        self.bias = bias

        self.num_kv = self.num_heads // self.kv_repeat
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** (-0.5)

        out_dim = self.dim + 2 * self.num_kv * self.dim // self.num_heads
        self.in_proj = nnx.Linear(
            in_features=self.dim, out_features=out_dim, use_bias=self.bias, rngs=rngs
        )
        self.out_proj = nnx.Linear(
            in_features=self.dim, out_features=self.dim, use_bias=self.bias, rngs=rngs
        )

        self.rope = None
        # TODO: Implement RoPE embeddings

    def __call__(self, xs: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        assert self.kv_repeat == 1, "Not implemented yet"

        B, T, HD = xs.shape
        qkv = self.in_proj(xs).reshape(B, T, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        # TODO: add RoPE embeddings

        k_len = k.shape[2]
        k_target_len = T + min(self.context, k_len - T)
        if k_target_len < k_len:
            k = k[:, :, k_len - k_target_len :]
            v = v[:, :, k_len - k_target_len :]

        q = q / self.scale
        xs = nnx.dot_product_attention(query=q, key=k, value=v, mask=mask)
        xs = xs.transpose(0, 2, 1, 3).reshape(B, T, HD)
        xs = self.out_proj(xs)

        return xs


class GatedMLP(nnx.Module):
    def __init__(self, dim: int, mlp_dim: int, bias: bool, rngs: nnx.Rngs):
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.hidden_dim = 2 * self.mlp_dim // 3
        if self.mlp_dim == 4 * self.dim:
            self.hidden_dim = 11 * self.dim // 4

        self.input_linear = nnx.Linear(
            in_features=self.dim,
            out_features=2 * self.hidden_dim,
            use_bias=bias,
            rngs=rngs,
        )
        self.output_linear = nnx.Linear(
            in_features=self.hidden_dim, out_features=self.dim, use_bias=bias, rngs=rngs
        )

    def __call__(self, xs: jax.Array) -> jax.Array:
        xs = self.input_linear(xs)
        B, T, _ = xs.shape
        xs = xs.reshape(B, T, 2, -1)
        return self.output_linear(jax.nn.silu(xs[:, :, 0]) * xs[:, :, 1])


class MLP(nnx.Module):
    def __init__(self, dim: int, mlp_dim: int, bias: bool, rngs: nnx.Rngs):
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.input_linear = nnx.Linear(
            in_features=self.dim,
            out_features=self.mlp_dim,
            use_bias=bias,
            rngs=rngs,
        )
        self.output_linear = nnx.Linear(
            in_features=self.mlp_dim, out_features=self.dim, use_bias=bias, rngs=rngs
        )

    def __call__(self, xs: jax.Array) -> jax.Array:
        return self.output_linear(jax.nn.gelu(self.input_linear(xs), approximate=True))


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_heads: int,
        kv_repeat: int,
        context: int,
        norm: str,
        bias_attn: bool,
        bias_mlp: bool,
        gating: bool,
        layer_scale: float,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.mlp_dim = mlp_dim

        # MLP
        if gating:
            self.gating = GatedMLP(
                dim=self.dim, mlp_dim=self.mlp_dim, bias=bias_mlp, rngs=rngs
            )
        else:
            self.gating = MLP(
                dim=self.dim, mlp_dim=self.mlp_dim, bias=bias_mlp, rngs=rngs
            )

        # Normalization
        if norm == "layer_norm":
            self.norm1 = nnx.LayerNorm(num_features=self.dim, epsilon=1e-5, rngs=rngs)
            self.norm2 = nnx.LayerNorm(num_features=self.dim, epsilon=1e-5, rngs=rngs)
        elif norm == "rms_norm":
            self.norm1 = nnx.RMSNorm(num_features=self.dim, epsilon=1e-8, rngs=rngs)
            self.norm2 = nnx.RMSNorm(num_features=self.dim, epsilon=1e-8, rngs=rngs)
        else:
            raise ValueError(f"Unknown normalization type: {norm}")

        # Layer Scale
        if layer_scale is not None:
            self.layer_scale_1 = LayerScale(dim=self.dim)
            self.layer_scale_2 = LayerScale(dim=self.dim)
        else:
            self.layer_scale_1 = Identity()
            self.layer_scale_2 = Identity()

        # Attention
        self.self_attn = Attention(
            num_heads=num_heads,
            kv_repeat=kv_repeat,
            dim=self.dim,
            context=context,
            bias=bias_attn,
            rngs=rngs,
        )

    def __call__(self, xs: jax.Array) -> jax.Array:
        n1 = self.norm1(xs)
        n1 = self.self_attn(n1)
        xs = xs + self.layer_scale_1(n1)
        xs = xs + self.layer_scale_2(self.gating(self.norm2(xs)))
        return xs


class Transformer(nnx.Module):
    def __init__(self, config: ml_collections.ConfigDict, rngs: nnx.Rngs):
        self.config = config
        self.layers = [
            TransformerLayer(
                dim=self.config.dim,
                mlp_dim=self.config.mlp_dim,
                num_heads=self.config.num_heads,
                kv_repeat=self.config.kv_repeat,
                context=self.config.context,
                norm=self.config.norm,
                bias_attn=self.config.bias_attn,
                bias_mlp=self.config.bias_mlp,
                gating=self.config.gating,
                layer_scale=self.config.layer_scale,
                rngs=rngs,
            )
            for _ in range(self.config.num_layers)
        ]

    def __call__(self, xs: jax.Array) -> jax.Array:
        for layer in self.layers:
            xs = layer(xs)
        return xs
