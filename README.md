# sam3-onnx

[ONNX](https://onnx.ai/) export and inference for [SAM3](https://github.com/facebookresearch/sam3).

## Quick start

```sh
git clone https://github.com/wkentaro/sam3-onnx.git --recursive && cd sam3-onnx

uvx hf download wkentaro/sam3-onnx-models-v0.4.0 --local-dir models  # download pre-exported models

uv run --no-dev infer_onnx.py --image images/bus.jpg --text-prompt person
uv run --no-dev infer_onnx.py --image images/sofa.jpg --text-prompt sofa
uv run --no-dev infer_onnx.py --image images/dog.jpg --box-prompt  # opens a gui to select box
uv run --no-dev infer_onnx.py --image images/bus.jpg --point-prompt 0.154,0.607  # detect instances similar to the pointed one
uv run --no-dev infer_onnx.py --image images/bus.jpg --interactive-point-prompt "0.154,0.607;0.154,0.787,0"  # segment the pointed instance (label 1=fg, 0=bg)
```

<img src="assets/example_bus_person.jpg" width="25%" /> <img src="assets/example_sofa_sofa.jpg" width="30%" /> <img src="assets/example_dog_face_boxprompt.jpg" width="36%" />

## Installation

```sh
git clone https://github.com/wkentaro/sam3-onnx.git --recursive && cd sam3-onnx
uv sync  # dev mode for exporting onnx unlike --no-dev
```

## Usage

**Inference with pytorch**

```sh
# with text prompt
uv run infer_torch.py --image images/bus.jpg --text-prompt person

# with box prompt
uv run infer_torch.py --image images/bus.jpg --box-prompt 0.548,0.658,0.043,0.095
uv run infer_torch.py --image images/bus.jpg --box-prompt  # opens a gui to select box

# with point prompt (exemplar: detect instances similar to the pointed one)
uv run infer_torch.py --image images/bus.jpg --point-prompt 0.154,0.607

# with interactive point prompt (segment the pointed instance, label 1=fg, 0=bg)
uv run infer_torch.py --image images/bus.jpg --interactive-point-prompt "0.154,0.607;0.154,0.787,0"
```

**Export to onnx**

```sh
# create models/*.onnx
uv run export_onnx.py

# upload to Hugging Face Hub
# uvx hf upload wkentaro/sam3-onnx-models-v0.4.0 models/ --include '*.onnx*'
```

**Inference with onnx**

```sh
# with text prompt
uv run infer_onnx.py --image images/bus.jpg --text-prompt person

# with box prompt
uv run infer_onnx.py --image images/bus.jpg --box-prompt 0.548,0.658,0.043,0.095
uv run infer_onnx.py --image images/bus.jpg --box-prompt  # opens a gui to select box

# with point prompt (exemplar: detect instances similar to the pointed one)
uv run infer_onnx.py --image images/bus.jpg --point-prompt 0.154,0.607

# with interactive point prompt (segment the pointed instance, label 1=fg, 0=bg)
uv run infer_onnx.py --image images/bus.jpg --interactive-point-prompt "0.154,0.607;0.154,0.787,0"
```

## Pre-exported ONNX models

If don't want to export yourself, download them from the [Hugging Face repo](https://huggingface.co/wkentaro/sam3-onnx-models-v0.4.0):

```
models
├── sam3_decoder.onnx
├── sam3_image_encoder.onnx
├── sam3_image_encoder.onnx.data
├── sam3_interactive_decoder.onnx
├── sam3_language_encoder.onnx
└── sam3_language_encoder.onnx.data
```

## ONNX model I/O

**sam3_image_encoder.onnx**

- Inputs: `image` uint8 `(3, 1008, 1008)` (RGB).
- Outputs: `vision_pos_enc_0..2`, `backbone_fpn_0..2` (SAM3 features for
  `sam3_decoder.onnx`), `high_res_feat_0` `(1, 32, 288, 288)`,
  `high_res_feat_1` `(1, 64, 144, 144)`, `image_embed` `(1, 256, 72, 72)`
  (SAM2 features for `sam3_interactive_decoder.onnx`).

**sam3_language_encoder.onnx**

- Inputs: `tokens` int64 `(1, 32)` (CLIP BPE tokens).
- Outputs: `text_attention_mask`, `text_memory`, `text_embeds`.

**sam3_decoder.onnx** (concept grounding: finds all matching instances)

- Inputs: image/language features above, plus geometric exemplar prompts:
  - `box_coords` float32 `(num_boxes, 1, 4)`: normalized `cx, cy, w, h`.
  - `box_labels` int64 `(num_boxes, 1)`: 1 positive, 0 negative.
  - `box_masks` bool `(1, num_boxes)`: True masks the box out.
  - `point_coords` float32 `(num_points, 1, 2)`: normalized `x, y`.
  - `point_labels` int64 `(num_points, 1)`: 1 positive, 0 negative.
  - `point_masks` bool `(1, num_points)`: True masks the point out.
  - `num_boxes` and `num_points` are dynamic. Pass a single masked-out
    dummy (as `infer_onnx.py` does) when a prompt type is unused.
- Outputs: `boxes` (normalized xyxy), `scores`, `masks` (probabilities at
  model native resolution; resize to the image size and threshold at 0.5).

**sam3_interactive_decoder.onnx** (SAM2-style: segments the pointed instance)

- Inputs:
  - `image_embed`, `high_res_feat_0`, `high_res_feat_1` from the image encoder.
  - `point_coords` float32 `(1, num_points, 2)`: normalized `x, y`.
  - `point_labels` int64 `(1, num_points)`: 1 foreground, 0 background,
    2/3 box left-top/right-bottom corners.
- Outputs: `mask_logits` `(1, 3, 288, 288)` (multimask; resize to the image
  size and threshold at 0), `mask_scores` `(1, 3)` (predicted mask IoU;
  pick the argmax mask).

## Examples

```sh
uv run infer_onnx.py --image images/bus.jpg --text-prompt person
uv run infer_onnx.py --image images/bus.jpg --text-prompt window
uv run infer_onnx.py --image images/bus.jpg --box-prompt 0.547,0.656,0.042,0.091  # select a block
```

<img src="assets/example_bus_person.jpg" width="30%" /> <img src="assets/example_bus_window.jpg" width="30%" /> <img src="assets/example_bus_block_boxprompt.jpg" width="30%" />

```sh
uv run infer_onnx.py --image images/sofa.jpg --text-prompt person
uv run infer_onnx.py --image images/sofa.jpg --text-prompt sofa
```

<img src="assets/example_sofa_person.jpg" width="40%" /> <img src="assets/example_sofa_sofa.jpg" width="40%" />

```sh
uv run infer_onnx.py --image images/dog.jpg --text-prompt dog
uv run infer_onnx.py --image images/dog.jpg --text-prompt ground
uv run infer_onnx.py --image images/dog.jpg --box-prompt 0.561,0.342,0.167,0.217
```

<img src="assets/example_dog_dog.jpg" width="30%" /> <img src="assets/example_dog_ground.jpg" width="30%" /> <img src="assets/example_dog_face_boxprompt.jpg" width="30%" />
