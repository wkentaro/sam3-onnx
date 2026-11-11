# SAM3-LiteText

[SAM3-LiteText](https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext)
replaces the language encoder with a distilled MobileCLIP encoder. Export it on
CPU using an upstream checkout. The released weights download automatically.
The encoders use MobileCLIP architectures and may count as derivatives under
Apple's [model license](https://github.com/apple/ml-mobileclip/blob/main/LICENSE_MODELS),
which limits use to non-commercial research. Check it before commercial use.

```sh
git clone --branch sam3_litetext https://github.com/SimonZeng7108/efficientsam3.git ../efficientsam3
git -C ../efficientsam3 checkout bef17f5c24dc5ef19dc1d8e9663345a2ae7f2f5a
uv run export_onnx.py --litetext s0 --litetext-source ../efficientsam3
uv run --no-dev infer_onnx.py --image images/bus.jpg --text-prompt person --litetext s0
```

Use `s0`, `s1`, or `l` for MobileCLIP-S0, MobileCLIP-S1, or MobileCLIP2-L.
Export creates `models/sam3_litetext_<variant>_language_encoder.onnx` and reuses
the existing image encoder and decoder. To use downloaded weights, add
`--litetext-checkpoint /path/to/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt`
to the export command. The upstream checkout and PyTorch are only needed for
export. The default export and inference commands use the original encoder.

The LiteText input is `tokens`, int64 with shape `1x16`, using the same CLIP
tokenizer. Prompts longer than 16 tokens, including the start and end tokens,
are truncated with the end token preserved. Outputs are `text_attention_mask`
with shape `1x32`, `text_memory` with shape `32x1x256`, and `text_embeds` with
shape `32x1x512` for S0/S1 or `32x1x768` for L. The last 16 positions are zero
padded and masked so existing 32-position decoders work without re-export.
The decoder uses the mask and memory only.
