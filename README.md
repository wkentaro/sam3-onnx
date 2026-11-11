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

**Export to onnx**

```sh
# create models/*.onnx, with models/sam3_*_fused.onnx for CUDA and WebGPU
uv run export_onnx.py

# or convert models from an older export in place, as export_onnx.py does
uv run postprocess_onnx.py models/*.onnx && rm models/*.onnx.data

# write float16 copies for GPUs, as models/*_fp16.onnx
uv run postprocess_onnx.py --fp16 models/*.onnx

# write an int8 image encoder for CPUs, as models/sam3_image_encoder_int8.onnx
uv run postprocess_onnx.py --int8 models/sam3_image_encoder.onnx

```

**SAM3-LiteText**

[SAM3-LiteText](docs/litetext.md) swaps in a distilled MobileCLIP language encoder. Its license may limit it to non-commercial research.

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

Add `--int8` with `--device cpu` for an int8 image encoder, which takes 40% less time on an x86 CPU. See [docs/performance.md](docs/performance.md#int8).

**Video tracking with onnx**

```sh
# find and track every person from --frame-idx through the video, both ways
uv run infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --text-prompt person --output /tmp/bedroom.mp4
# track the object at a point or box on --frame-idx through the video, both ways
uv run infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --point-prompt 0.22,0.65 --output /tmp/bedroom.mp4
uv run infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --box-prompt 0.417,0.37,0.208,0.741 --output /tmp/bedroom.mp4
```

The v0.6.0 models predate video tracking. Create the tracker models and the updated decoders and memory encoder with `uv run export_onnx.py`.

**Inference with onnx on GPU**

```sh
UV_PROJECT_ENVIRONMENT=.venv-gpu uv run --no-dev --no-group cpu --group gpu infer_onnx.py --image images/bus.jpg --text-prompt person --device cuda
```

Add `--fp16` to run the float16 models, nearly twice as fast on a GPU. See [docs/performance.md](docs/performance.md#float16).

The `gpu` group installs `onnxruntime-gpu` with its CUDA and cuDNN libraries, so no system CUDA install is needed. Keep it in its own environment without the dev group: onnxruntime and onnxruntime-gpu overwrite each other's files, and so do the cuDNN libraries of torch and onnxruntime-gpu.

**Inference with onnx on Apple GPU (macOS)**

```sh
uv run --no-dev --group webgpu infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --text-prompt person --device coreml --fp16 --output /tmp/bedroom.mp4
```

`--device coreml` runs the image encoder through CoreML and the other models on WebGPU. See [docs/performance.md](docs/performance.md#coreml).

**Inference with onnx on WebGPU (macOS, Linux, Windows)**

```sh
uv run --no-dev --group webgpu infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --text-prompt person --device webgpu --output /tmp/bedroom.mp4
```

The `webgpu` group installs osam's WebGPU plugin, `onnxruntime-ep-webgpu`, which runs every model on the GPU through Vulkan, Direct3D 12, or Metal. On macOS, prefer `--device coreml` above.

**Compare speed with PyTorch**

```sh
uv run benchmark.py --runtime torch
UV_PROJECT_ENVIRONMENT=.venv-gpu uv run --no-dev --no-group cpu --group gpu benchmark.py --device cuda
```

`benchmark.py` prints the mean time of each model and of the whole, per image or tracked frame. See [docs/performance.md](docs/performance.md#benchmark).

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

## Development

See [docs/development.md](docs/development.md) for PyTorch inference, the parity checks against PyTorch, and uploading models, and [docs/video-api.md](docs/video-api.md) for the Python API of video tracking.
