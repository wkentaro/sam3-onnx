# sam3-onnx

[ONNX](https://onnx.ai/) export and inference for [SAM3](https://github.com/facebookresearch/sam3).

## Quick start

```sh
git clone https://github.com/wkentaro/sam3-onnx.git --recursive && cd sam3-onnx

uvx hf download wkentaro/sam3-onnx-models-v0.6.0 --local-dir models  # download pre-exported models

uv run --no-dev infer_onnx.py --image images/bus.jpg --text-prompt person
uv run --no-dev infer_onnx.py --image images/sofa.jpg --text-prompt sofa
uv run --no-dev infer_onnx.py --image images/dog.jpg --box-prompt  # opens a gui to select box
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

# with several example boxes: cx,cy,w,h[,label] (label 0 for a negative example), also with text
uv run infer_torch.py --image images/bus.jpg --box-prompt 0.184,0.601,0.24,0.466 --box-prompt 0.35,0.585,0.146,0.419,0
uv run infer_torch.py --image images/bus.jpg --text-prompt person --box-prompt 0.35,0.585,0.146,0.419,0

# with point prompt: x,y[,label] (normalized, label 0 for background), repeatable
uv run infer_torch.py --image images/bus.jpg --point-prompt 0.18,0.55
uv run infer_torch.py --image images/bus.jpg --point-prompt 0.18,0.55 --point-prompt 0.2,0.7 --single-mask  # one mask, for several points
```

**Export to onnx**

```sh
# create models/*.onnx
uv run export_onnx.py

# or convert models from an older export in place, as export_onnx.py does
uv run postprocess_onnx.py models/*.onnx && rm models/*.onnx.data

# upload to Hugging Face Hub
# uvx hf upload wkentaro/sam3-onnx-models-v0.6.0 models/ --include '*.onnx'
```

**SAM3-LiteText**

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

**Inference with onnx**

```sh
# with text prompt
uv run infer_onnx.py --image images/bus.jpg --text-prompt person

# with box prompt
uv run infer_onnx.py --image images/bus.jpg --box-prompt 0.548,0.658,0.043,0.095
uv run infer_onnx.py --image images/bus.jpg --box-prompt  # opens a gui to select box

# with several example boxes: cx,cy,w,h[,label] (label 0 for a negative example), also with text
uv run infer_onnx.py --image images/bus.jpg --box-prompt 0.184,0.601,0.24,0.466 --box-prompt 0.35,0.585,0.146,0.419,0
uv run infer_onnx.py --image images/bus.jpg --text-prompt person --box-prompt 0.35,0.585,0.146,0.419,0

# with point prompt: x,y[,label] (normalized, label 0 for background), repeatable
uv run infer_onnx.py --image images/bus.jpg --point-prompt 0.18,0.55
uv run infer_onnx.py --image images/bus.jpg --point-prompt 0.18,0.55 --point-prompt 0.2,0.7 --single-mask  # one mask, for several points
```

**Video tracking with onnx**

```sh
# find and track every person from --frame-idx through the video, both ways
uv run infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --text-prompt person --output /tmp/bedroom.mp4
# track the object at a point or box on --frame-idx through the video, both ways
uv run infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --point-prompt 0.22,0.65 --output /tmp/bedroom.mp4
uv run infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --box-prompt 0.417,0.37,0.208,0.741 --output /tmp/bedroom.mp4
```

`video_onnx.VideoInference` ports SAM3's video predictor (`Sam3VideoInferenceWithInstanceInteractivity`). A text prompt finds the objects on every frame with `sam3_decoder.onnx`, and a `Tracker` tracks each one. Points then refine an object or add one, and `remove_object` drops one.

`track_onnx.Tracker` ports SAM3's video tracker (`Sam3TrackerPredictor`) to numpy on top of four models: `sam3_point_decoder.onnx`, `sam3_memory_encoder.onnx`, `sam3_memory_attention.onnx`, and `sam3_mask_downsampler.onnx`. It takes point, box, and mask prompts for several objects, refinement clicks on any frame, forward and reverse propagation, and object removal, with the same method names as the PyTorch predictor.

The v0.6.0 models predate video tracking. Create the tracker models and the updated decoders and memory encoder with `uv run export_onnx.py`.

**Inference with onnx on GPU**

```sh
UV_PROJECT_ENVIRONMENT=.venv-gpu uv run --no-dev --no-group cpu --group gpu infer_onnx.py --image images/bus.jpg --text-prompt person --device cuda
```

The `gpu` group installs `onnxruntime-gpu` with its CUDA and cuDNN libraries, so no system CUDA install is needed. Keep it in its own environment without the dev group: onnxruntime and onnxruntime-gpu overwrite each other's files, and so do the cuDNN libraries of torch and onnxruntime-gpu.

**Inference with onnx on Apple GPU (macOS)**

```sh
uv run --no-dev infer_onnx.py --image images/bus.jpg --text-prompt person --device coreml
```

The image encoder runs on the GPU through CoreML, over twice as fast as on CPU (2.5 s against 6 s on an M5). The other models stay on CPU, where they take under 2 s.

**Inference with onnx on WebGPU (macOS, Linux, Windows)**

```sh
uv run --no-dev --group webgpu infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --text-prompt person --device webgpu --output /tmp/bedroom.mp4
```

The `webgpu` group installs osam's WebGPU plugin, `onnxruntime-ep-webgpu`, which runs every model on the GPU through Vulkan, Direct3D 12, or Metal. For video, use it over CoreML: CoreML can't compile the tracker's memory attention, whose input grows as frames fill the memory, so a frame takes 5.8 s against 6.7 s on an M5.

**Check text-prompt scores against PyTorch on CPU**

Save the apple image from [issue #2](https://github.com/wkentaro/sam3-onnx/issues/2), then run:

```sh
uv run test_text_parity.py --apple-image /path/to/apple.jpg
```

The check compares preprocessing, image and language features, and decoder scores for apples and `images/bus.jpg` with "person". It also feeds PyTorch features to the ONNX decoder to isolate decoder errors. Both pipelines must return the same detection count and sorted scores within 0.01. PIL bilinear resizing can differ from PyTorch's tensor resize by one uint8 level. This inference fix works with the existing models; no re-export or Hugging Face upload is needed.

**Check video tracking against PyTorch on CUDA**

```sh
uv run test_video_parity.py
uv run test_video_inference_parity.py
```

The check runs point, refinement, reverse, box, mask, two-object, and removal cases on the first 60 frames of `bedroom.mp4` with the PyTorch tracker in float32, and fails if a frame's mask IoU drops below 0.99.

`test_video_inference_parity.py` checks `VideoInference` the same way against `build_sam3_video_model()`. It runs text and box prompts, forward and reverse, point refinement, an added object, and removal. It fails if the object ids differ, a mask IoU drops below 0.99, or a score differs by more than 0.01.

**Compare speed with PyTorch**

```sh
uv run benchmark.py --runtime torch
UV_PROJECT_ENVIRONMENT=.venv-gpu uv run --no-dev --no-group cpu --group gpu benchmark.py --device cuda
```

`benchmark.py` runs text, box, and point prompts on `images/bus.jpg`, then point, box, and text prompts through 40 frames of `bedroom.mp4`. It prints the mean time of each model and of the whole in milliseconds, per image or per tracked frame. Each case runs once untimed first. A frame's time runs from the JPEG files of the video to the masks of its last frame, over the frame count, so it holds loading the frames in both runtimes. Compare the totals across runtimes: a model's time on the PyTorch side is that of the nearest module, which does less than the ONNX model. `--runtime torch` runs upstream as it ships, in bfloat16 on CUDA. `--device` takes the same values as `infer_onnx.py`.

## Pre-exported ONNX models

If don't want to export yourself, download them from the [Hugging Face repo](https://huggingface.co/wkentaro/sam3-onnx-models-v0.6.0):

```
models
├── sam3_decoder.onnx
├── sam3_image_encoder.onnx
├── sam3_language_encoder.onnx
└── sam3_point_decoder.onnx
```

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
