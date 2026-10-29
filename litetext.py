import importlib.util
import pathlib
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from torch.nn import functional as F


class LiteTextLanguageEncoder(torch.nn.Module):
    def __init__(self, encoder: Any, projector: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeds = self.encoder.forward_embedding(tokens)
        memory = self.projector(
            self.encoder(embeds, return_all_tokens=True, input_is_embeddings=True)
        )
        # Existing decoders expect 32 positions; the released students use 16.
        return (
            F.pad(tokens == 0, (0, 16), value=True),
            F.pad(memory, (0, 0, 0, 16)).transpose(0, 1),
            F.pad(embeds, (0, 0, 0, 16)).transpose(0, 1),
        )


def load_litetext_encoder(
    source: pathlib.Path, variant: str, checkpoint: pathlib.Path | None = None
) -> LiteTextLanguageEncoder:
    module_path = source / "sam3/sam3/backbones/mobile_clip.py"
    spec = importlib.util.spec_from_file_location("litetext_mobile_clip", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LiteText encoder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dim = 768 if variant == "l" else 512
    cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "dim": dim,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": dim // 64,
        "n_transformer_layers": 4 if variant == "s0" else 12,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "mct" if variant == "s0" else "base",
    }
    model = LiteTextLanguageEncoder(
        module.MobileCLIPTextTransformer(cfg=cfg, projection_dim=dim),
        torch.nn.Linear(dim, 256),
    )
    if checkpoint is None:
        name = "mobileclip2_l" if variant == "l" else f"mobileclip_{variant}"
        checkpoint = pathlib.Path(
            hf_hub_download(
                "Simon7108528/EfficientSAM3",
                f"sam3_litetext/efficient_sam3_image_encoder_{name}_ctx16.pt",
            )
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model", state)
    prefix = "backbone.language_backbone."
    weights = {
        key.removeprefix("detector.").removeprefix(prefix): value
        for key, value in state.items()
        if key.removeprefix("detector.").startswith(prefix)
    }
    model.load_state_dict(weights, strict=True)
    model.encoder.resize_pos_embed(16)
    return model.eval()
