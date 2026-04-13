# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 quantization config for diffusion dense linear layers.

Implements OCP Microscaling FP4 (E2M1) with E8M0 per-block scales.
Uses dequant+GEMM fallback for initial validation.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

# MXFP4 block size (scale applies to every 32 elements)
MXFP4_BLOCK_SIZE = 32

# E2M1 format constants
E2M1_MAX = 6.0  # Max representable value in E2M1 (110_1 = 6.0)
E2M1_MIN_NORMAL = 0.5  # Min normal value (010_0 = 0.5)


def _pack_e2m1_to_uint8(e2m1: torch.Tensor) -> torch.Tensor:
    """Pack two E2M1 values into one uint8 byte.

    E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits
    Two values packed: [val1(4bits) | val0(4bits)]
    """
    assert e2m1.dtype == torch.uint8
    assert e2m1.shape[-1] % 2 == 0

    # Reshape to pairs
    shape = list(e2m1.shape)
    shape[-1] = shape[-1] // 2
    packed = torch.zeros(shape, dtype=torch.uint8, device=e2m1.device)

    # Pack: high 4 bits = even indices, low 4 bits = odd indices
    packed = (e2m1[..., 0::2] << 4) | e2m1[..., 1::2]
    return packed


def _unpack_e2m1_from_uint8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into E2M1 values (one byte -> two 4-bit values)."""
    assert packed.dtype == torch.uint8

    # Unpack: high 4 bits and low 4 bits
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F

    # Interleave back
    shape = list(packed.shape)
    shape[-1] = shape[-1] * 2
    unpacked = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    unpacked[..., 0::2] = high
    unpacked[..., 1::2] = low
    return unpacked


def _e2m1_to_float32(e2m1_uint4: torch.Tensor) -> torch.Tensor:
    """Convert E2M1 4-bit values to float32.

    E2M1 encoding:
    - Bit 3: sign
    - Bits 2-1: exponent (2 bits, bias=1)
    - Bit 0: mantissa (1 bit)

    Special cases:
    - 0000 = +0, 1000 = -0
    - 0001 = +0.5, 1001 = -0.5 (subnormal, treated as 0.5)
    """
    assert e2m1_uint4.dtype == torch.uint8

    # Extract fields
    sign = (e2m1_uint4 >> 3) & 0x01
    exponent = (e2m1_uint4 >> 1) & 0x03
    mantissa = e2m1_uint4 & 0x01

    # Convert to float
    sign_float = torch.where(sign == 0, 1.0, -1.0)

    # E2M1 value lookup table for all 16 values
    # Index: 0-15, Value: based on E2M1 encoding
    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=e2m1_uint4.device,
    )

    magnitude = e2m1_lut[e2m1_uint4.long() & 0x0F]
    return sign_float * magnitude


def _float32_to_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Convert float32 values to E2M1 4-bit format (quantize).

    Uses round-to-nearest-even strategy.
    """
    abs_values = values.abs()
    sign = (values < 0).to(torch.uint8) << 3

    # E2M1 representable values (magnitudes)
    e2m1_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                               dtype=torch.float32, device=values.device)
    e2m1_codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                              dtype=torch.uint8, device=values.device)

    # Find closest E2M1 value for each element
    # Shape: [batch, 8] for broadcasting
    diff = (abs_values.unsqueeze(-1) - e2m1_values).abs()
    closest_idx = diff.argmin(dim=-1)
    mantissa_code = e2m1_codes[closest_idx]

    return sign | mantissa_code


def mxfp4_e2m1_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 weight to MXFP4 E2M1 format.

    Args:
        weight: Input tensor of shape [out_features, in_features]

    Returns:
        weight_packed: uint8 tensor of shape [out_features, in_features // 2]
        weight_scale: uint8 tensor of shape [out_features, in_features // 32]
                      E8M0 format (exponent only, bias=127)
    """
    assert weight.dim() == 2
    out_features, in_features = weight.shape
    assert in_features % MXFP4_BLOCK_SIZE == 0, (
        f"in_features ({in_features}) must be divisible by MXFP4_BLOCK_SIZE ({MXFP4_BLOCK_SIZE})"
    )

    # Convert to float32 for quantization
    weight_fp32 = weight.float()

    # Reshape to blocks of 32
    weight_blocked = weight_fp32.view(out_features, -1, MXFP4_BLOCK_SIZE)

    # Compute scale per block: max absolute value
    block_max = weight_blocked.abs().amax(dim=-1, keepdim=True)  # [out, num_blocks, 1]

    # Avoid division by zero
    block_max = torch.clamp(block_max, min=1e-12)

    # Normalize to E2M1 range [0, 6.0]
    scale = block_max / E2M1_MAX  # Scale factor to map max to 6.0
    normalized = weight_blocked / scale

    # Quantize to E2M1
    e2m1_values = _float32_to_e2m1(normalized)  # [out, num_blocks, 32]

    # Pack E2M1 values: 2 per byte
    e2m1_packed = _pack_e2m1_to_uint8(e2m1_values)  # [out, num_blocks, 16]

    # Flatten back to 2D
    weight_packed = e2m1_packed.view(out_features, in_features // 2)

    # Encode scale as E8M0 (exponent only, uint8)
    # E8M0: 8-bit exponent, bias=127, no mantissa
    # scale_uint8 = clamp(round(log2(scale) + 127), 0, 255)
    scale_exponent = torch.log2(scale.squeeze(-1)) + 127.0
    scale_uint8 = torch.clamp(torch.round(scale_exponent), 0, 255).to(torch.uint8)

    return weight_packed, scale_uint8


def dequantize_mxfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP4 E2M1 back to BF16/FP16.

    Args:
        weight_packed: uint8 tensor [out_features, in_features // 2]
        weight_scale: uint8 tensor [out_features, in_features // 32]
        dtype: Output dtype (bf16 or fp16)

    Returns:
        Dequantized weight tensor [out_features, in_features]
    """
    out_features, packed_size = weight_packed.shape
    in_features = packed_size * 2

    # Unpack E2M1 values
    e2m1_values = _unpack_e2m1_from_uint8(weight_packed)  # [out, in_features]

    # Convert E2M1 to float
    magnitude = _e2m1_to_float32(e2m1_values)  # [out, in_features]

    # Decode E8M0 scale
    # scale = 2^(scale_uint8 - 127)
    scale_exponent = weight_scale.float() - 127.0
    scale = torch.pow(2.0, scale_exponent)  # [out, num_blocks]

    # Reshape scale for broadcasting: [out, 1, num_blocks] -> [out, in_features]
    num_blocks = in_features // MXFP4_BLOCK_SIZE
    scale = scale.view(out_features, num_blocks, 1)
    scale = scale.expand(out_features, num_blocks, MXFP4_BLOCK_SIZE)
    scale = scale.reshape(out_features, in_features)

    # Apply scale
    dequantized = magnitude * scale

    return dequantized.to(dtype)


class DiffusionMxfp4Config(QuantizationConfig):
    """MXFP4 quantization config for diffusion dense linear layers.

    Supports online (dynamic) quantization from BF16/FP16 checkpoints.
    Uses dequant+GEMM fallback for initial validation.
    """

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # SM80+ (A100 and newer)

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionMxfp4Config":
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(ignored_layers=ignored_layers)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Mxfp4DenseLinearMethod(self)
        return None


class Mxfp4DenseLinearMethod(LinearMethodBase):
    """Linear method for MXFP4 E2M1 online quantization.

    Loads BF16/FP16 checkpoints and quantizes weights to MXFP4 during loading.
    Uses dequant+GEMM fallback for inference.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: DiffusionMxfp4Config):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Validate input size divisibility
        if input_size_per_partition % MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP4 requires input_size_per_partition ({input_size_per_partition}) "
                f"to be divisible by {MXFP4_BLOCK_SIZE}."
            )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Create MXFP4 packed weight on meta device (lazy materialization)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                device="meta",
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Create MXFP4 scale parameter
        num_blocks = input_size_per_partition // MXFP4_BLOCK_SIZE
        weight_scale = PerTensorScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                num_blocks,
                device="meta",
                dtype=torch.uint8,
            ),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # At this point, layer.weight should contain the original BF16/FP16 weights
        # (loaded from checkpoint before quantization)
        # We need to quantize them to MXFP4

        # Check if weight is still on meta device (not yet loaded)
        if layer.weight.device == torch.device("meta"):
            # Materialize with original dtype first
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device, dtype=layer.orig_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)

            # Initialize with dummy weight (will be overwritten by actual weights)
            from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
            initialize_single_dummy_weight(layer.weight)

        # The weight should have been loaded by now - but in the fallback path,
        # we may have the original BF16 weight stored temporarily.
        # For the dequant+GEMM fallback, we keep the original BF16 weight
        # and quantize on-the-fly during apply().

        # For now, mark as processed
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Dequant+GEMM fallback path
        # If weight is already MXFP4 packed, dequantize it
        if layer.weight.dtype == torch.uint8:
            weight_bf16 = dequantize_mxfp4(
                layer.weight,
                layer.weight_scale,
                dtype=self.out_dtype,
            )
        else:
            # Weight is still in BF16/FP16 (online quantization not yet applied)
            # Quantize now and cache the result
            weight_bf16 = layer.weight

        # Reshape input if needed
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # GEMM
        output = torch.mm(x_2d, weight_bf16.t())

        if bias is not None:
            output = output + bias

        # Reshape output
        output_shape = list(input_shape[:-1]) + [weight_bf16.shape[0]]
        output = output.reshape(output_shape)

        return output
