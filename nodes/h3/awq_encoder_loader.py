# Copyright (c) 2026 Fred Bliss (fredbliss / fbjr). All credit to the original author.
# Vendored from: https://huggingface.co/fbjr/qwen3-vl-32b-W4A16-AWQ-H3
#   (comfyui_minimax_h3_awq_loader.py), included in comfyui_dagthomas with attribution.
# The generated file below is unmodified.
## GENERATED FILE - DO NOT EDIT BY HAND.
# Source: ComfyUI-h3-explorations/h3_awq_encoder.py
# Source SHA-256: 3f7c09df442e9fc736c01a5f3abe4e4104072c761baa802763587fef340e4854
# Runtime config SHA-256 values are recorded in _EMBEDDED_CONFIG_SHA256.

"""Load a compatible compressed-tensors AWQ Qwen3-VL checkpoint as H3.

This is deliberately a custom loader instead of a patch to ``CLIPLoader``.
ComfyUI natively supplies the H3 architecture and tokenizer (including the
seven H3 tokens), and comfy-kitchen supplies the CUDA W4A16 operator.  Core
does not currently recognize compressed-tensors' Hugging Face namespace,
packing, or metadata.  This generated module is the standalone adapter for
that gap.

Adaptation is in-memory and view-based for the 4-bit weights; it does not
write a second multi-gigabyte checkpoint.  Its four small runtime configs are
embedded from the versioned ComfyUI-h3-explorations snapshot.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import sys
import types
from pathlib import Path

from comfy_api.latest import ComfyExtension, io

logger = logging.getLogger(__name__)

_EMBEDDED_CONFIG_TEXT = {
    'config.json': '{\n  "architectures": [\n    "Qwen3VLForConditionalGeneration"\n  ],\n  "dtype": "bfloat16",\n  "image_token_id": 151655,\n  "model_type": "qwen3_vl",\n  "quantization_config": {\n    "config_groups": {\n      "group_0": {\n        "format": "pack-quantized",\n        "input_activations": null,\n        "output_activations": null,\n        "targets": [\n          "Linear"\n        ],\n        "weights": {\n          "actorder": null,\n          "block_structure": null,\n          "dynamic": false,\n          "group_size": 128,\n          "num_bits": 4,\n          "observer": "memoryless_minmax",\n          "observer_kwargs": {},\n          "scale_dtype": null,\n          "strategy": "group",\n          "symmetric": true,\n          "type": "int",\n          "zp_dtype": null\n        }\n      }\n    },\n    "format": "pack-quantized",\n    "global_compression_ratio": null,\n    "ignore": [\n      "model.visual.blocks.0.attn.qkv",\n      "model.visual.blocks.0.attn.proj",\n      "model.visual.blocks.0.mlp.linear_fc1",\n      "model.visual.blocks.0.mlp.linear_fc2",\n      "model.visual.blocks.1.attn.qkv",\n      "model.visual.blocks.1.attn.proj",\n      "model.visual.blocks.1.mlp.linear_fc1",\n      "model.visual.blocks.1.mlp.linear_fc2",\n      "model.visual.blocks.2.attn.qkv",\n      "model.visual.blocks.2.attn.proj",\n      "model.visual.blocks.2.mlp.linear_fc1",\n      "model.visual.blocks.2.mlp.linear_fc2",\n      "model.visual.blocks.3.attn.qkv",\n      "model.visual.blocks.3.attn.proj",\n      "model.visual.blocks.3.mlp.linear_fc1",\n      "model.visual.blocks.3.mlp.linear_fc2",\n      "model.visual.blocks.4.attn.qkv",\n      "model.visual.blocks.4.attn.proj",\n      "model.visual.blocks.4.mlp.linear_fc1",\n      "model.visual.blocks.4.mlp.linear_fc2",\n      "model.visual.blocks.5.attn.qkv",\n      "model.visual.blocks.5.attn.proj",\n      "model.visual.blocks.5.mlp.linear_fc1",\n      "model.visual.blocks.5.mlp.linear_fc2",\n      "model.visual.blocks.6.attn.qkv",\n      "model.visual.blocks.6.attn.proj",\n      "model.visual.blocks.6.mlp.linear_fc1",\n      "model.visual.blocks.6.mlp.linear_fc2",\n      "model.visual.blocks.7.attn.qkv",\n      "model.visual.blocks.7.attn.proj",\n      "model.visual.blocks.7.mlp.linear_fc1",\n      "model.visual.blocks.7.mlp.linear_fc2",\n      "model.visual.blocks.8.attn.qkv",\n      "model.visual.blocks.8.attn.proj",\n      "model.visual.blocks.8.mlp.linear_fc1",\n      "model.visual.blocks.8.mlp.linear_fc2",\n      "model.visual.blocks.9.attn.qkv",\n      "model.visual.blocks.9.attn.proj",\n      "model.visual.blocks.9.mlp.linear_fc1",\n      "model.visual.blocks.9.mlp.linear_fc2",\n      "model.visual.blocks.10.attn.qkv",\n      "model.visual.blocks.10.attn.proj",\n      "model.visual.blocks.10.mlp.linear_fc1",\n      "model.visual.blocks.10.mlp.linear_fc2",\n      "model.visual.blocks.11.attn.qkv",\n      "model.visual.blocks.11.attn.proj",\n      "model.visual.blocks.11.mlp.linear_fc1",\n      "model.visual.blocks.11.mlp.linear_fc2",\n      "model.visual.blocks.12.attn.qkv",\n      "model.visual.blocks.12.attn.proj",\n      "model.visual.blocks.12.mlp.linear_fc1",\n      "model.visual.blocks.12.mlp.linear_fc2",\n      "model.visual.blocks.13.attn.qkv",\n      "model.visual.blocks.13.attn.proj",\n      "model.visual.blocks.13.mlp.linear_fc1",\n      "model.visual.blocks.13.mlp.linear_fc2",\n      "model.visual.blocks.14.attn.qkv",\n      "model.visual.blocks.14.attn.proj",\n      "model.visual.blocks.14.mlp.linear_fc1",\n      "model.visual.blocks.14.mlp.linear_fc2",\n      "model.visual.blocks.15.attn.qkv",\n      "model.visual.blocks.15.attn.proj",\n      "model.visual.blocks.15.mlp.linear_fc1",\n      "model.visual.blocks.15.mlp.linear_fc2",\n      "model.visual.blocks.16.attn.qkv",\n      "model.visual.blocks.16.attn.proj",\n      "model.visual.blocks.16.mlp.linear_fc1",\n      "model.visual.blocks.16.mlp.linear_fc2",\n      "model.visual.blocks.17.attn.qkv",\n      "model.visual.blocks.17.attn.proj",\n      "model.visual.blocks.17.mlp.linear_fc1",\n      "model.visual.blocks.17.mlp.linear_fc2",\n      "model.visual.blocks.18.attn.qkv",\n      "model.visual.blocks.18.attn.proj",\n      "model.visual.blocks.18.mlp.linear_fc1",\n      "model.visual.blocks.18.mlp.linear_fc2",\n      "model.visual.blocks.19.attn.qkv",\n      "model.visual.blocks.19.attn.proj",\n      "model.visual.blocks.19.mlp.linear_fc1",\n      "model.visual.blocks.19.mlp.linear_fc2",\n      "model.visual.blocks.20.attn.qkv",\n      "model.visual.blocks.20.attn.proj",\n      "model.visual.blocks.20.mlp.linear_fc1",\n      "model.visual.blocks.20.mlp.linear_fc2",\n      "model.visual.blocks.21.attn.qkv",\n      "model.visual.blocks.21.attn.proj",\n      "model.visual.blocks.21.mlp.linear_fc1",\n      "model.visual.blocks.21.mlp.linear_fc2",\n      "model.visual.blocks.22.attn.qkv",\n      "model.visual.blocks.22.attn.proj",\n      "model.visual.blocks.22.mlp.linear_fc1",\n      "model.visual.blocks.22.mlp.linear_fc2",\n      "model.visual.blocks.23.attn.qkv",\n      "model.visual.blocks.23.attn.proj",\n      "model.visual.blocks.23.mlp.linear_fc1",\n      "model.visual.blocks.23.mlp.linear_fc2",\n      "model.visual.blocks.24.attn.qkv",\n      "model.visual.blocks.24.attn.proj",\n      "model.visual.blocks.24.mlp.linear_fc1",\n      "model.visual.blocks.24.mlp.linear_fc2",\n      "model.visual.blocks.25.attn.qkv",\n      "model.visual.blocks.25.attn.proj",\n      "model.visual.blocks.25.mlp.linear_fc1",\n      "model.visual.blocks.25.mlp.linear_fc2",\n      "model.visual.blocks.26.attn.qkv",\n      "model.visual.blocks.26.attn.proj",\n      "model.visual.blocks.26.mlp.linear_fc1",\n      "model.visual.blocks.26.mlp.linear_fc2",\n      "model.visual.merger.linear_fc1",\n      "model.visual.merger.linear_fc2",\n      "model.visual.deepstack_merger_list.0.linear_fc1",\n      "model.visual.deepstack_merger_list.0.linear_fc2",\n      "model.visual.deepstack_merger_list.1.linear_fc1",\n      "model.visual.deepstack_merger_list.1.linear_fc2",\n      "model.visual.deepstack_merger_list.2.linear_fc1",\n      "model.visual.deepstack_merger_list.2.linear_fc2",\n      "lm_head"\n    ],\n    "kv_cache_scheme": null,\n    "quant_method": "compressed-tensors",\n    "quantization_status": "compressed",\n    "sparsity_config": {},\n    "transform_config": {},\n    "version": "0.18.1.a20260821"\n  },\n  "text_config": {\n    "attention_bias": false,\n    "attention_dropout": 0.0,\n    "bos_token_id": 151643,\n    "dtype": "bfloat16",\n    "eos_token_id": 151645,\n    "head_dim": 128,\n    "hidden_act": "silu",\n    "hidden_size": 5120,\n    "initializer_range": 0.02,\n    "intermediate_size": 25600,\n    "max_position_embeddings": 262144,\n    "model_type": "qwen3_vl_text",\n    "num_attention_heads": 64,\n    "num_hidden_layers": 64,\n    "num_key_value_heads": 8,\n    "pad_token_id": null,\n    "rms_norm_eps": 1e-06,\n    "rope_parameters": {\n      "mrope_interleaved": true,\n      "mrope_section": [\n        24,\n        20,\n        20\n      ],\n      "rope_theta": 5000000,\n      "rope_type": "default"\n    },\n    "use_cache": true,\n    "vocab_size": 151936\n  },\n  "tie_word_embeddings": false,\n  "transformers_version": "5.15.1",\n  "video_token_id": 151656,\n  "vision_config": {\n    "deepstack_visual_indexes": [\n      8,\n      16,\n      24\n    ],\n    "depth": 27,\n    "dtype": "bfloat16",\n    "hidden_act": "gelu_pytorch_tanh",\n    "hidden_size": 1152,\n    "in_channels": 3,\n    "initializer_range": 0.02,\n    "intermediate_size": 4304,\n    "model_type": "qwen3_vl_vision",\n    "num_heads": 16,\n    "num_position_embeddings": 2304,\n    "out_hidden_size": 5120,\n    "patch_size": 16,\n    "spatial_merge_size": 2,\n    "temporal_patch_size": 2\n  },\n  "vision_end_token_id": 151653,\n  "vision_start_token_id": 151652\n}\n',
    'tokenizer_config.json': '{\n  "add_prefix_space": false,\n  "backend": "tokenizers",\n  "bos_token": null,\n  "clean_up_tokenization_spaces": false,\n  "eos_token": "<|im_end|>",\n  "errors": "replace",\n  "extra_special_tokens": [\n    "<|im_start|>",\n    "<|im_end|>",\n    "<|object_ref_start|>",\n    "<|object_ref_end|>",\n    "<|box_start|>",\n    "<|box_end|>",\n    "<|quad_start|>",\n    "<|quad_end|>",\n    "<|vision_start|>",\n    "<|vision_end|>",\n    "<|vision_pad|>",\n    "<|image_pad|>",\n    "<|video_pad|>",\n    "<d>",\n    "</d>",\n    "<|cutoff|>",\n    "<|lyrics_start|>",\n    "<|lyrics_end|>",\n    "<|caption_start|>",\n    "<|caption_end|>"\n  ],\n  "is_local": true,\n  "local_files_only": false,\n  "max_pixels": 301056,\n  "min_pixels": 200704,\n  "model_max_length": 262144,\n  "pad_token": "<|endoftext|>",\n  "processor_class": "Qwen3VLProcessor",\n  "split_special_tokens": false,\n  "tokenizer_class": "Qwen2Tokenizer",\n  "unk_token": null\n}\n',
    'processor_config.json': '{\n  "image_processor": {\n    "do_convert_rgb": true,\n    "do_normalize": true,\n    "do_rescale": true,\n    "do_resize": true,\n    "image_mean": [\n      0.5,\n      0.5,\n      0.5\n    ],\n    "image_processor_type": "Qwen2VLImageProcessor",\n    "image_std": [\n      0.5,\n      0.5,\n      0.5\n    ],\n    "merge_size": 2,\n    "patch_size": 16,\n    "resample": 3,\n    "rescale_factor": 0.00392156862745098,\n    "size": {\n      "longest_edge": 301056,\n      "shortest_edge": 200704\n    },\n    "temporal_patch_size": 2\n  },\n  "processor_class": "Qwen3VLProcessor",\n  "video_processor": {\n    "do_convert_rgb": true,\n    "do_normalize": true,\n    "do_rescale": true,\n    "do_resize": true,\n    "do_sample_frames": true,\n    "fps": 2,\n    "image_mean": [\n      0.5,\n      0.5,\n      0.5\n    ],\n    "image_std": [\n      0.5,\n      0.5,\n      0.5\n    ],\n    "max_frames": 768,\n    "merge_size": 2,\n    "min_frames": 4,\n    "patch_size": 16,\n    "resample": 3,\n    "rescale_factor": 0.00392156862745098,\n    "return_metadata": false,\n    "size": {\n      "longest_edge": 25165824,\n      "shortest_edge": 4096\n    },\n    "temporal_patch_size": 2,\n    "video_processor_type": "Qwen3VLVideoProcessor"\n  }\n}\n',
    'video_preprocessor_config.json': '{\n    "size": {\n        "longest_edge": 25165824,\n        "shortest_edge": 4096\n    },\n    "patch_size": 16,\n    "temporal_patch_size": 2,\n    "merge_size": 2,\n    "image_mean": [\n        0.5,\n        0.5,\n        0.5\n    ],\n    "image_std": [\n        0.5,\n        0.5,\n        0.5\n    ],\n    "processor_class": "Qwen3VLProcessor",\n    "video_processor_type": "Qwen3VLVideoProcessor"\n}\n',
}

_EMBEDDED_CONFIG_SHA256 = {
    'config.json': '9917eaef4fa12e25b213103f0e6593a8b967e96517f56ad01a9b003227cafcc5',
    'tokenizer_config.json': 'ed52146b25a3a8e50b2fb84341fce37188d21b6ef44c071619a6a3e0baca051a',
    'processor_config.json': '62be31450acaccdf8ef9a528e31a1d511918db85132bc9b8b708f254311306a0',
    'video_preprocessor_config.json': '00bd47a5eaaf8760744a12658cd99ba168b818edc4c0a983b90157289c8e546a',
}
CONFIG_SOURCE = (
    'four runtime configs embedded by build_h3_awq_standalone.py '
    '(f0bb54a302466b23d7acd4a403cac796d696d2d6bd0b7ec4281aab951f3bcf27)'
)
QUANT_FORMAT = "h3_awq_w4a16"
H3_LAYERS = 50
SOURCE_LAYERS = 64
GROUP_SIZE = 128
EXPECTED_QUANTIZED_LINEARS = H3_LAYERS * 7

_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


@functools.lru_cache(maxsize=None)
def _config(name: str) -> dict:
    try:
        text = _EMBEDDED_CONFIG_TEXT[name]
    except KeyError as exc:
        raise FileNotFoundError(
            f"{name} is not embedded in this standalone loader"
        ) from exc
    return json.loads(text)

def _quant_contract() -> dict:
    cfg = _config("config.json")
    group = ((cfg.get("quantization_config") or {}).get("config_groups") or {}).get(
        "group_0", {}
    )
    weights = group.get("weights") or {}
    text = cfg.get("text_config") or {}
    required = {
        "format": group.get("format"),
        "bits": weights.get("num_bits"),
        "group_size": weights.get("group_size"),
        "symmetric": weights.get("symmetric"),
        "strategy": weights.get("strategy"),
        "dtype": cfg.get("dtype"),
        "layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
    }
    expected = {
        "format": "pack-quantized", "bits": 4, "group_size": GROUP_SIZE,
        "symmetric": True, "strategy": "group", "dtype": "bfloat16",
        "layers": SOURCE_LAYERS, "hidden_size": 5120,
    }
    if required != expected:
        raise ValueError(
            "vendored encoder config is not the W4A16 H3 contract this "
            f"adapter implements: got {required!r}, expected {expected!r}"
        )
    return cfg


def source_video_pixel_bounds() -> tuple[int, int]:
    """Return the selected encoder artifact's declared video pixel budget."""
    size = _config("video_preprocessor_config.json").get("size") or {}
    lo, hi = size.get("shortest_edge"), size.get("longest_edge")
    if not isinstance(lo, int) or not isinstance(hi, int) or not 0 < lo < hi:
        raise ValueError(f"source video processor has invalid size bounds: {size!r}")
    return lo, hi


def source_video_patch_geometry() -> dict:
    """Return patch/normalization settings from the encoder's own snapshot."""
    cfg = _config("video_preprocessor_config.json")
    keys = ("patch_size", "temporal_patch_size", "merge_size",
            "image_mean", "image_std")
    geometry = {key: cfg[key] for key in keys if key in cfg}
    if set(geometry) != set(keys):
        raise ValueError(
            "source video processor is missing patch or normalization settings"
        )
    return geometry


def _validate_metadata(metadata: dict | None) -> None:
    metadata = metadata or {}
    if metadata.get("scheme") != "w4a16" or metadata.get("quantization") != "awq":
        raise ValueError(
            "checkpoint is not the expected AWQ W4A16 artifact: safetensors "
            f"metadata says scheme={metadata.get('scheme')!r}, "
            f"quantization={metadata.get('quantization')!r}"
        )
    raw = metadata.get("config")
    if not isinstance(raw, str):
        raise ValueError("checkpoint safetensors metadata has no embedded config")
    try:
        embedded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint embedded config is not valid JSON") from exc
    if embedded != _quant_contract():
        raise ValueError(
            "checkpoint embedded config differs from the versioned config "
            f"snapshot from {CONFIG_SOURCE}"
        )


def _drop_source_key(name: str) -> bool:
    """True for full-Qwen tensors H3 intentionally does not consume."""
    match = _LAYER.match(name)
    if match and int(match.group(1)) >= H3_LAYERS:
        return True
    return name.startswith("lm_head.") or name.startswith("model.language_model.norm.")


def adapt_compressed_state_dict(state_dict: dict, metadata: dict | None) -> dict:
    """Destructively adapt the raw 64-layer HF state dict to Comfy H3.

    compressed-tensors packs eight signed int4 values into each int32 after
    adding eight.  On little-endian hosts an int8 view yields four consecutive
    bytes, each already containing the two unsigned nibbles comfy-kitchen
    expects.  No weight-sized unpack or copy is needed.
    """
    import torch

    _validate_metadata(metadata)
    if sys.byteorder != "little":
        raise RuntimeError("the zero-copy AWQ repack is only defined on little-endian hosts")

    shapes = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight_shape") and not _drop_source_key(name):
            prefix = name[:-len(".weight_shape")]
            shape = tuple(int(x) for x in tensor.tolist())
            if len(shape) != 2:
                raise ValueError(f"{name} declares non-matrix shape {shape!r}")
            shapes[prefix] = shape

    out = {}
    quantized = set()
    for source_name in list(state_dict):
        tensor = state_dict.pop(source_name)
        if _drop_source_key(source_name):
            continue

        name = source_name
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        elif name.startswith("model.visual."):
            name = "visual." + name[len("model.visual."):]

        if source_name.endswith(".weight_shape"):
            continue
        if source_name.endswith(".weight_packed"):
            source_prefix = source_name[:-len(".weight_packed")]
            logical = shapes.get(source_prefix)
            if tensor.dtype != torch.int32 or logical is None:
                raise ValueError(
                    f"{source_name} needs int32 storage and a weight_shape companion"
                )
            expected = (logical[0], logical[1] // 8)
            if tuple(tensor.shape) != expected or logical[1] % GROUP_SIZE:
                raise ValueError(
                    f"{source_name} shape {tuple(tensor.shape)} does not encode "
                    f"declared {logical} at 4-bit/group-{GROUP_SIZE}"
                )
            target_prefix = name[:-len(".weight_packed")]
            qweight = tensor.view(torch.int8)
            if tuple(qweight.shape) != (logical[0], logical[1] // 2):
                raise AssertionError("int32-to-int8 view did not preserve packed rows")
            out[f"{target_prefix}.weight"] = qweight
            conf = {"format": QUANT_FORMAT, "group_size": GROUP_SIZE}
            out[f"{target_prefix}.comfy_quant"] = torch.tensor(
                list(json.dumps(conf, sort_keys=True).encode("utf-8")),
                dtype=torch.uint8,
            )
            quantized.add(target_prefix)
            continue
        if source_name.endswith(".weight_scale"):
            source_prefix = source_name[:-len(".weight_scale")]
            logical = shapes.get(source_prefix)
            if tensor.dtype != torch.bfloat16 or logical is None:
                raise ValueError(f"{source_name} has no usable BF16 scale/shape pair")
            expected = (logical[0], logical[1] // GROUP_SIZE)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{source_name} shape {tuple(tensor.shape)} != {expected}"
                )
            target_prefix = name[:-len(".weight_scale")]
            # kitchen owns scales as (K/group, N); compressed-tensors stores
            # (N, K/group). CUDA consumes flat row-major data, so materialize
            # the transpose once rather than handing it a strided view.
            out[f"{target_prefix}.weight_scale"] = tensor.t().contiguous()
            continue

        out[name] = tensor

    if len(quantized) != EXPECTED_QUANTIZED_LINEARS:
        raise ValueError(
            f"adapted {len(quantized)} quantized linears; H3 needs "
            f"{EXPECTED_QUANTIZED_LINEARS} (7 in each of {H3_LAYERS} layers)"
        )
    missing_scales = [p for p in quantized if f"{p}.weight_scale" not in out]
    if missing_scales:
        raise ValueError(f"quantized linears missing scales: {missing_scales[:3]}")
    if "visual.deepstack_merger_list.0.norm.weight" not in out:
        raise ValueError("adapted checkpoint has no Qwen3-VL DeepStack vision tower")
    if "model.layers.49.self_attn.q_proj.weight" not in out:
        raise ValueError("adapted checkpoint does not reach H3 layer 49")
    if any(k.startswith("model.layers.50.") for k in out):
        raise AssertionError("full-Qwen layer 50 escaped the H3 truncation")

    logger.info(
        "[h3-awq] adapted compressed-tensors checkpoint in memory: %d "
        "W4A16 linears, first %d/%d language layers, BF16 vision/embedding",
        len(quantized), H3_LAYERS, SOURCE_LAYERS,
    )
    return out


def _install_quant_format():
    """Register the kitchen layout only when this custom loader executes."""
    import torch
    import comfy.ops
    import comfy.quant_ops
    from comfy_kitchen.tensor import TensorCoreAWQW4A16Layout  # noqa: F401

    spec = {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "weight_zeros"},
        "comfy_tensor_layout": "TensorCoreAWQW4A16Layout",
        "quantize_input": False,
    }
    existing = comfy.quant_ops.QUANT_ALGOS.get(QUANT_FORMAT)
    if existing is not None and existing != spec:
        raise RuntimeError(
            f"ComfyUI already registered {QUANT_FORMAT!r} with a different contract"
        )
    comfy.quant_ops.QUANT_ALGOS[QUANT_FORMAT] = spec
    comfy.ops.QUANT_ALGOS[QUANT_FORMAT] = spec
    return spec


@functools.lru_cache(maxsize=1)
def awq_operations():
    """Comfy mixed ops with one local load branch for symmetric AWQ W4A16."""
    import torch
    import comfy.ops
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreAWQW4A16Layout

    spec = _install_quant_format()
    base = comfy.ops.mixed_precision_ops(
        compute_dtype=torch.bfloat16, full_precision_mm=False
    )

    class H3AWQOperations(base):
        class Linear(base.Linear):
            def _forward(self, input, weight, bias):
                if (isinstance(weight, QuantizedTensor)
                        and weight._layout_cls == "TensorCoreAWQW4A16Layout"):
                    # SDClipModel intentionally constructs and forwards H3
                    # embeddings as FP32. AWQ is W4A16: kitchen's fused CUDA
                    # backend accepts BF16/FP16 activation, while an FP32 x
                    # silently selects the eager dequantization backend. Cast
                    # only across the quantized matmul, then restore the
                    # caller's dtype for residual arithmetic.
                    output_dtype = input.dtype
                    kernel_dtype = weight._params.scale.dtype
                    input = input.to(dtype=kernel_dtype)
                    if bias is not None:
                        bias = bias.to(dtype=kernel_dtype)

                    if input.device.type == "cuda" and not getattr(
                            H3AWQOperations, "_awq_backend_logged", False):
                        from comfy_kitchen.registry import registry
                        backend = registry.get_capable_backend(
                            "gemv_awq_w4a16",
                            kwargs={
                                "x": input,
                                "qweight": weight._qdata,
                                "wscales": weight._params.scale,
                                "wzeros": weight._params.zeros,
                                "bias": bias,
                                "group_size": weight._params.group_size,
                            },
                        )
                        logger.info(
                            "[h3-awq] W4A16 dispatch backend=%s, "
                            "activation %s -> %s",
                            backend, output_dtype, kernel_dtype,
                        )
                        H3AWQOperations._awq_backend_logged = True

                    return torch.nn.functional.linear(
                        input, weight, bias
                    ).to(dtype=output_dtype)
                return super()._forward(input, weight, bias)

            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                conf_key = f"{prefix}comfy_quant"
                conf_tensor = state_dict.get(conf_key)
                conf = None
                if conf_tensor is not None:
                    conf = json.loads(conf_tensor.numpy().tobytes())
                if not conf or conf.get("format") != QUANT_FORMAT:
                    return super()._load_from_state_dict(
                        state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs,
                    )

                weight_key = f"{prefix}weight"
                scale_key = f"{prefix}weight_scale"
                zeros_key = f"{prefix}weight_zeros"
                weight = state_dict.pop(weight_key, None)
                scale = state_dict.pop(scale_key, None)
                zeros = state_dict.pop(zeros_key, None)
                state_dict.pop(conf_key, None)
                if weight is None or scale is None:
                    raise ValueError(
                        f"{prefix.rstrip('.')} is missing AWQ weight or scale"
                    )
                if tuple(weight.shape) != (
                    self._orig_shape[0], self._orig_shape[1] // 2
                ):
                    raise ValueError(
                        f"{prefix} packed shape {tuple(weight.shape)} does not "
                        f"match linear {self._orig_shape}"
                    )

                device = self.factory_kwargs["device"]
                dtype = self.factory_kwargs["dtype"]
                weight = weight.to(device=device, dtype=spec["storage_t"])
                scale = scale.to(device=device, dtype=dtype)
                if zeros is None:
                    # The source config is symmetric, so the affine zero term
                    # is exactly zero. kitchen's general AWQ ABI still takes
                    # the tensor explicitly.
                    zeros = torch.zeros_like(scale, device=device, dtype=dtype)
                else:
                    zeros = zeros.to(device=device, dtype=dtype)

                params = TensorCoreAWQW4A16Layout.Params(
                    scale=scale, zeros=zeros,
                    group_size=int(conf.get("group_size", GROUP_SIZE)),
                    transposed=False, orig_dtype=dtype,
                    orig_shape=self._orig_shape,
                )
                self.quant_format = QUANT_FORMAT
                self.layout_type = spec["comfy_tensor_layout"]
                self._full_precision_mm_config = False
                self.weight = torch.nn.Parameter(
                    QuantizedTensor(weight, self.layout_type, params),
                    requires_grad=False,
                )

                # Let torch load the ordinary bias, then erase the missing
                # report for the weight we deliberately consumed ourselves.
                torch.nn.Module._load_from_state_dict(
                    self, state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs,
                )
                for key in (weight_key, scale_key, zeros_key, conf_key):
                    if key in missing_keys:
                        missing_keys.remove(key)

    H3AWQOperations.__name__ = "H3AWQOperations"
    return H3AWQOperations


@functools.lru_cache(maxsize=1)
def _image_processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    return Qwen2VLImageProcessor(
        **_config("processor_config.json")["image_processor"]
    )


def _source_image_patches(images, device):
    """Run the source checkpoint's declared still-image processor."""
    import torch

    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
        raise ValueError(f"Qwen image must be [B,H,W,3], got {tuple(images.shape)}")
    image = images[0].detach().permute(2, 0, 1).to("cpu")
    if image.is_floating_point():
        image = image.mul(255).round().clamp_(0, 255).to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    batch = _image_processor().preprocess(image, return_tensors="pt")
    return batch["pixel_values"], batch["image_grid_thw"].to(device=device)


def _source_video_block_patches(frames, device):
    """Patchify an already duration-fitted two-frame Qwen video block."""
    import torch

    cfg = _config("video_preprocessor_config.json")
    temporal = int(cfg["temporal_patch_size"])
    patch = int(cfg["patch_size"])
    merge = int(cfg["merge_size"])
    if (frames.ndim != 4 or frames.shape[-1] != 3 or
            frames.shape[0] != temporal):
        raise ValueError(
            f"Qwen video block must be [{temporal},H,W,3], got {tuple(frames.shape)}"
        )
    height, width = int(frames.shape[1]), int(frames.shape[2])
    factor = patch * merge
    if height % factor or width % factor:
        raise ValueError(
            f"Qwen video block {width}x{height} was not fitted to the source "
            f"processor's {factor}-pixel grid. Use "
            "MiniMaxH3ReferenceConditioning video_policy='encoder' or 'release'."
        )

    imgs = frames.permute(0, 3, 1, 2)
    mean = torch.tensor(cfg["image_mean"], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["image_std"], device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std
    grid_h, grid_w = height // patch, width // patch
    patches = imgs.reshape(
        1, temporal, 3, grid_h // merge, merge, patch,
        grid_w // merge, merge, patch,
    ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(
        grid_h * grid_w, 3 * temporal * patch * patch
    )
    grid = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.long)
    return flatten, grid


def install_source_processors(clip) -> None:
    """Bind source-config preprocessing to this CLIP instance only."""
    import torch

    model = clip.cond_stage_model.qwen3vl_32b.transformer

    def preprocess_embed(this, embed, device):
        if embed.get("type") != "image":
            return None, None
        if embed.get("minimax_video_block", False):
            flatten, grid = _source_video_block_patches(embed["data"], device)
        else:
            flatten, grid = _source_image_patches(embed["data"], device)
        merged, deepstack = this.visual(
            flatten.to(device=device, dtype=torch.float32), grid
        )
        return merged, {"grid": grid, "deepstack": deepstack}

    model.preprocess_embed = types.MethodType(preprocess_embed, model)
    model._h3_processor_source = CONFIG_SOURCE


def _validate_loaded_state_contract(clip, provided_shapes: dict[str, tuple]) -> None:
    """Reject missing, extra, or shape-incompatible adapted model tensors.

    Core intentionally loads text encoders with ``strict=False``. That is a
    useful general policy, but unsafe for this format adapter: an absent
    ordinary vision/norm weight otherwise leaves a factory-created parameter
    behind and the loader still returns a CLIP. Compare the adapted inventory
    with the concrete native H3 module that core selected.

    Symmetric AWQ is the single intentional exception. The source omits affine
    zero tensors and ``H3AWQOperations`` constructs exact zeros while loading.
    """
    model = clip.cond_stage_model.qwen3vl_32b.transformer
    expected_state = model.state_dict()
    expected = set(expected_state)
    provided = set(provided_shapes)
    quantized = {
        name[:-len(".comfy_quant")]
        for name in provided
        if name.endswith(".comfy_quant")
    }
    synthesized = {f"{prefix}.weight_zeros" for prefix in quantized}
    missing = sorted(expected - provided - synthesized)
    unexpected = sorted(provided - expected)
    mismatched = []
    for name in sorted(expected & provided):
        # This byte tensor serializes configuration rather than model data.
        # Presence is part of the inventory; its decoded format/group values
        # were validated while adapting and loading the quantized linear.
        if name.endswith(".comfy_quant"):
            continue
        actual = tuple(provided_shapes[name])
        wanted = tuple(expected_state[name].shape)
        if actual != wanted:
            mismatched.append((name, actual, wanted))
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:3]}")
        raise ValueError(
            "selected checkpoint does not exactly populate the native H3 "
            "architecture after adaptation: " + "; ".join(details)
        )


def _validate_native_tokenizer(clip) -> None:
    """Prove native Comfy's tokenizer realizes the snapshotted token list."""
    cfg = _config("tokenizer_config.json")
    declared = cfg.get("extra_special_tokens") or []
    tokenizer = clip.tokenizer.qwen3vl_32b.tokenizer
    vocab = tokenizer.get_vocab()
    if len(declared) != 20:
        raise ValueError(
            f"source encoder declares {len(declared)} special tokens, expected 20"
        )
    if len(set(declared)) != len(declared):
        raise ValueError("source encoder declares duplicate special tokens")
    expected = {
        **{token: 151644 + index for index, token in enumerate(declared[:13])},
        **{token: 151669 + index for index, token in enumerate(declared[13:])},
    }
    actual = {token: vocab.get(token) for token in declared}
    if actual != expected:
        raise ValueError(
            "native ComfyUI tokenizer token ids disagree with the selected "
            f"encoder config: got {actual}, expected {expected}"
        )
    cfg = _config("config.json")
    declared_roles = {
        "<|vision_start|>": cfg.get("vision_start_token_id"),
        "<|vision_end|>": cfg.get("vision_end_token_id"),
        "<|image_pad|>": cfg.get("image_token_id"),
        "<|video_pad|>": cfg.get("video_token_id"),
    }
    role_mismatches = {
        token: (token_id, expected[token])
        for token, token_id in declared_roles.items()
        if token_id != expected[token]
    }
    if role_mismatches:
        raise ValueError(
            "source config token roles disagree with tokenizer_config: "
            f"{role_mismatches} (config, tokenizer)"
        )


def _load_clip(path: str, embedding_directory, device: str = "default",
               disable_dynamic: bool = False, install_cache: bool = True):
    import torch
    import comfy.sd
    import comfy.utils

    state_dict, metadata = comfy.utils.load_torch_file(
        path, safe_load=True, return_metadata=True
    )
    state_dict = adapt_compressed_state_dict(state_dict, metadata)
    provided_shapes = {name: tuple(tensor.shape) for name, tensor in state_dict.items()}
    model_options = {"custom_operations": awq_operations()}
    if device == "cpu":
        cpu = torch.device("cpu")
        model_options["load_device"] = model_options["offload_device"] = cpu
    clip = comfy.sd.load_text_encoder_state_dicts(
        [state_dict], embedding_directory=embedding_directory,
        clip_type=comfy.sd.CLIPType.MINIMAX, model_options=model_options,
        disable_dynamic=disable_dynamic,
    )
    _validate_loaded_state_contract(clip, provided_shapes)
    install_source_processors(clip)
    _validate_native_tokenizer(clip)
    if install_cache:
        clip.patcher.cached_patcher_init = (
            load_h3_awq_model_patcher,
            (path, embedding_directory, device),
        )
    logger.info(
        "[h3-awq] loaded %s through standalone compressed-tensors adapter; "
        "architecture/tokenizer are native ComfyUI, W4A16 execution is "
        "comfy-kitchen, preprocessing is source-config driven",
        Path(path).name,
    )
    return clip


def load_h3_awq_model_patcher(path: str, embedding_directory,
                              device: str = "default", disable_dynamic=False):
    return _load_clip(
        path, embedding_directory, device=device,
        disable_dynamic=disable_dynamic, install_cache=False,
    ).patcher


class MiniMaxH3AWQEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        # Filename is not the format contract. Offer the directory's real
        # safetensors population and validate compressed-tensors AWQ metadata
        # after the user selects one; never manufacture a missing menu item.
        names = sorted(
            n for n in folder_paths.get_filename_list("text_encoders")
            if n.endswith(".safetensors")
        )
        return io.Schema(
            node_id="MiniMaxH3AWQEncoderLoader",
            display_name="Load MiniMax H3 Compressed-Tensors AWQ Encoder",
            category="MiniMaxH3/loaders",
            description=(
                "Standalone adapter for Qwen3-VL-32B H3 checkpoints using "
                "compressed-tensors W4A16 AWQ. ComfyUI supplies native H3 "
                "architecture/tokenizer; "
                "comfy-kitchen supplies W4A16 CUDA execution. This node "
                "converts compressed-tensors packing/metadata in memory and "
                "uses the source image/video processor configs. Core "
                "CLIPLoader only lists this file; it cannot load this format."
            ),
            inputs=[
                io.Combo.Input("encoder_name", options=names),
                io.Combo.Input(
                    "device", options=["default", "cpu"], default="default",
                    optional=True,
                ),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, encoder_name, device="default"):
        import folder_paths

        path = folder_paths.get_full_path_or_raise("text_encoders", encoder_name)
        clip = _load_clip(
            path, folder_paths.get_folder_paths("embeddings"), device=device
        )
        return io.NodeOutput(clip)


class MiniMaxH3AWQStandaloneExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AWQEncoderLoader]


async def comfy_entrypoint() -> MiniMaxH3AWQStandaloneExtension:
    return MiniMaxH3AWQStandaloneExtension()
