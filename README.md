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

Add `--int8` to run the int8 image encoder, which `uv run postprocess_onnx.py --int8 models/sam3_image_encoder.onnx` writes. Its linear layers compute in int8. On an i9-12900K it encodes an image in 4.3 s against 7.5 s, and on an M5 CPU in 6.1 s against 7.1 s. It gives the same text detections, with scores within 0.007, and its tracked masks have a mean IoU of 0.994 with float32's, as PyTorch has 0.993 between bfloat16 and float32. The other models stay float32: with int8 tracker models, one of two tracked objects was lost, and int8 decoders ran no faster. `--int8` works with `--device cpu` and also in `infer_video_onnx.py` and `benchmark.py`.

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

Add `--fp16` to run the float16 models, which `uv run postprocess_onnx.py --fp16 models/*.onnx` writes. On an RTX 3080 Ti they track a video frame in 229 ms against 400 ms, with the same detections and masks within the difference that PyTorch has between bfloat16 and float32. `--fp16` works with `--device cuda`, `webgpu`, and `coreml`, and leaves the models that run on the CPU in float32.

On `--device cuda` and `webgpu`, the image encoder's features stay on the GPU for the models that read them, as `infer_onnx.run_on_device` returns them. With `--fp16` on CUDA, that takes a text prompt on an image from 249 ms to 225 ms, as its 116 MB of features no longer go to the CPU and back.

On `--device cuda` and `webgpu`, the image encoder runs from `sam3_image_encoder_fused.onnx` when `export_onnx.py` has written it. Its RoPE swaps each channel pair with one Gather instead of splitting and stacking them, and its attention runs as onnxruntime's `MultiHeadAttention`. With `--fp16` on an RTX 3080 Ti, it encodes an image in 151 ms against 192 ms, and a text prompt on an image takes 181 ms, against 173 ms in PyTorch. On WebGPU it encodes in 459 ms against 900 ms. CoreML takes 6 minutes to compile it, so `--device coreml` and `cpu` run `sam3_image_encoder.onnx`.

On `--device cuda`, the memory attention runs from `sam3_memory_attention_fused.onnx`, whose attention is onnxruntime's `MultiHeadAttention` as well. The logits of its cross-attention grow with each frame, and onnxruntime's CUDA arena, which the sessions share, keeps a new region for them on every frame. Tracking a video with `sam3_memory_attention.onnx` alone runs out of memory on a 12 GB GPU, and with the fused copy it peaks at 6.5 GB. It costs 6 ms per frame for 3 objects, and runs 3 to 6 times as slow on the CPU and WebGPU, which keep the plain file. Its float16 copy takes the queries of each attention in two halves: the op runs 5184 queries in 81 blocks of 64, one more than the 80 SMs of an RTX 3080 Ti, which take a second pass for it, while a half has its keys split between the SMs. That takes it from 93 ms to 68 ms per frame for 3 objects, against 62 ms in PyTorch. In float32 the op runs another kernel, which halves slow down, so the float32 file keeps whole attentions.

The decoder's attention runs as `MultiHeadAttention` on every device, as CoreML never runs the decoder. With `--fp16` on CUDA it decodes in 22 ms against 34 ms, and on the CPU in 1088 ms against 1303 ms.

The `gpu` group installs `onnxruntime-gpu` with its CUDA and cuDNN libraries, so no system CUDA install is needed. Keep it in its own environment without the dev group: onnxruntime and onnxruntime-gpu overwrite each other's files, and so do the cuDNN libraries of torch and onnxruntime-gpu.

**Inference with onnx on Apple GPU (macOS)**

```sh
uv run --no-dev --group webgpu infer_video_onnx.py --video sam3/assets/videos/bedroom.mp4 --text-prompt person --device coreml --fp16 --output /tmp/bedroom.mp4
```

`--device coreml` runs the image encoder on the GPU through CoreML, 3 times as fast as WebGPU runs it. It also runs `sam3_memory_attention_fixed.onnx`, a copy of the tracker's memory attention padded to a fixed shape, in about half the time WebGPU takes for the one whose input grows as frames fill the memory. CoreML can't compile an input whose shape changes, so the other models run on WebGPU when the `webgpu` group is installed, and on the CPU without it. With `--fp16` on an M5, a text prompt on an image takes 1.2 s and a tracked video frame of one object takes 1.2 s.

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

`benchmark.py` runs text, box, and point prompts on `images/bus.jpg`, then point, box, and text prompts through 40 frames of `bedroom.mp4`. It prints the mean time of each model and of the whole in milliseconds, per image or per tracked frame. Each case runs once untimed first. A frame's time runs from the JPEG files of the video to the masks of its last frame, over the frame count, so it holds loading the frames in both runtimes. Compare the totals across runtimes: a model's time on the PyTorch side is that of the nearest module, which does less than the ONNX model. `--runtime torch` runs upstream as it ships, in bfloat16 on CUDA. `--device` takes the same values as `infer_onnx.py`. On WebGPU, a run returns before the GPU finishes the features it keeps on the GPU, so their time counts toward the model that reads them first; the totals hold.

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
