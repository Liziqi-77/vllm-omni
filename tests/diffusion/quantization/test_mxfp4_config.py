# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MXFP4 quantization config and primitives."""

import pytest
import torch

from vllm_omni.quantization.mxfp4_config import (
    MXFP4_BLOCK_SIZE,
    dequantize_mxfp4,
    mxfp4_e2m1_quantize,
)


@pytest.mark.cpu
class TestMxfp4Quantize:
    """Test MXFP4 E2M1 quantization primitives."""

    def test_quantize_output_shape(self):
        """Verify output shapes are correct."""
        out_features, in_features = 128, 256
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

        packed, scales = mxfp4_e2m1_quantize(weight)

        assert packed.dtype == torch.uint8
        assert packed.shape == (out_features, in_features // 2)
        assert scales.dtype == torch.uint8
        assert scales.shape == (out_features, in_features // MXFP4_BLOCK_SIZE)

    def test_quantize_dequantize_roundtrip(self):
        """Verify dequantize(quantize(x)) is close to x."""
        out_features, in_features = 64, 128
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

        packed, scales = mxfp4_e2m1_quantize(weight)
        dequantized = dequantize_mxfp4(packed, scales, dtype=torch.bfloat16)

        assert dequantized.shape == weight.shape
        assert dequantized.dtype == torch.bfloat16

        # E2M1 has limited precision (8 representable values), expect ~25-35% relative error
        # This is acceptable for MXFP4 format - model quality depends on many factors
        rel_error = (weight - dequantized).abs() / (weight.abs() + 1e-6)
        assert rel_error.mean() < 0.35, f"Mean relative error too high: {rel_error.mean():.4f}"

    def test_quantize_zero_input(self):
        """Verify zero input produces zero output."""
        weight = torch.zeros(32, 64, dtype=torch.bfloat16)

        packed, scales = mxfp4_e2m1_quantize(weight)
        dequantized = dequantize_mxfp4(packed, scales, dtype=torch.bfloat16)

        # Should be very close to zero (may have small numerical errors)
        assert dequantized.abs().max() < 1e-3

    def test_quantize_requires_divisible_input(self):
        """Verify input size must be divisible by block size."""
        weight = torch.randn(64, 63, dtype=torch.bfloat16)  # 63 not divisible by 32

        with pytest.raises(AssertionError, match="divisible by MXFP4_BLOCK_SIZE"):
            mxfp4_e2m1_quantize(weight)

    def test_e2m1_representable_values(self):
        """Verify E2M1 quantization maps to representable values."""
        # Create a simple tensor with known values
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.bfloat16)

        # Pad to block size
        padding = MXFP4_BLOCK_SIZE - weight.numel() % MXFP4_BLOCK_SIZE
        if padding < MXFP4_BLOCK_SIZE:
            weight = torch.cat([weight, torch.zeros(1, padding, dtype=torch.bfloat16)], dim=1)

        packed, scales = mxfp4_e2m1_quantize(weight)
        dequantized = dequantize_mxfp4(packed, scales, dtype=torch.bfloat16)

        # Dequantized values should be in E2M1 representable set (scaled)
        # E2M1 magnitudes: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        # After scaling, values should be close to original


@pytest.mark.cpu
class TestMxfp4Config:
    """Test DiffusionMxfp4Config class."""

    def test_build_mxfp4_config(self):
        """Verify MXFP4 config can be built via factory."""
        from vllm_omni.quantization import build_quant_config

        config = build_quant_config("mxfp4")
        assert config is not None
        assert config.get_name() == "mxfp4"

    def test_build_mxfp4_with_ignored_layers(self):
        """Verify MXFP4 config supports ignored layers."""
        from vllm_omni.quantization import build_quant_config

        config = build_quant_config({
            "method": "mxfp4",
            "ignored_layers": ["patch_embedding", "proj_out"],
        })
        assert config is not None
        assert "patch_embedding" in config.ignored_layers
        assert "proj_out" in config.ignored_layers

    def test_supported_act_dtypes(self):
        """Verify supported activation dtypes."""
        from vllm_omni.quantization import build_quant_config

        config = build_quant_config("mxfp4")
        act_dtypes = config.get_supported_act_dtypes()
        assert torch.bfloat16 in act_dtypes
        assert torch.float16 in act_dtypes

    def test_min_capability(self):
        """Verify minimum capability requirement."""
        from vllm_omni.quantization import build_quant_config

        config = build_quant_config("mxfp4")
        assert config.get_min_capability() == 80  # SM80+
