# Performance

How each device runs the models, and the measurements behind each choice.

## float16

Add `--fp16` to run the float16 models, which `uv run postprocess_onnx.py --fp16 models/*.onnx` writes. On an RTX 3080 Ti they track a video frame in 229 ms against 400 ms, with the same detections and masks within the difference that PyTorch has between bfloat16 and float32. `--fp16` works with `--device cuda`, `webgpu`, and `coreml`, and leaves the models that run on the CPU in float32.

## Features stay on the GPU

On `--device cuda` and `webgpu`, the image encoder's features stay on the GPU for the models that read them, as `infer_onnx.run_on_device` returns them. With `--fp16` on CUDA, that takes a text prompt on an image from 249 ms to 225 ms, as its 116 MB of features no longer go to the CPU and back.

## Fused image encoder

On `--device cuda` and `webgpu`, the image encoder runs from `sam3_image_encoder_fused.onnx` when `export_onnx.py` has written it. Its RoPE swaps each channel pair with one Gather instead of splitting and stacking them, and its attention runs as onnxruntime's `MultiHeadAttention`. With `--fp16` on an RTX 3080 Ti, it encodes an image in 151 ms against 192 ms, and a text prompt on an image takes 181 ms, against 173 ms in PyTorch. On WebGPU it encodes in 459 ms against 900 ms. CoreML takes 6 minutes to compile it, so `--device coreml` and `cpu` run `sam3_image_encoder.onnx`.

## Fused memory attention

On `--device cuda`, the memory attention runs from `sam3_memory_attention_fused.onnx`, whose attention is onnxruntime's `MultiHeadAttention` as well. The logits of its cross-attention grow with each frame, and onnxruntime's CUDA arena, which the sessions share, keeps a new region for them on every frame. Tracking a video with `sam3_memory_attention.onnx` alone runs out of memory on a 12 GB GPU, and with the fused copy it peaks at 6.5 GB. It costs 6 ms per frame for 3 objects, and runs 3 to 6 times as slow on the CPU and WebGPU, which keep the plain file. Its float16 copy takes the queries of each attention in two halves: the op runs 5184 queries in 81 blocks of 64, one more than the 80 SMs of an RTX 3080 Ti, which take a second pass for it, while a half has its keys split between the SMs. That takes it from 93 ms to 68 ms per frame for 3 objects, against 62 ms in PyTorch. In float32 the op runs another kernel, which halves slow down, so the float32 file keeps whole attentions.

## Fused decoder

The decoder's attention runs as `MultiHeadAttention` on every device, as CoreML never runs the decoder. With `--fp16` on CUDA it decodes in 22 ms against 34 ms, and on the CPU in 1088 ms against 1303 ms.

## CoreML

`--device coreml` runs the image encoder on the GPU through CoreML, 3 times as fast as WebGPU runs it. It also runs `sam3_memory_attention_fixed.onnx`, a copy of the tracker's memory attention padded to a fixed shape, in about half the time WebGPU takes for the one whose input grows as frames fill the memory. CoreML can't compile an input whose shape changes, so the other models run on WebGPU when the `webgpu` group is installed, and on the CPU without it. With `--fp16` on an M5, a text prompt on an image takes 1.2 s and a tracked video frame of one object takes 1.2 s.

## int8

Add `--int8` to run the int8 image encoder, which `uv run postprocess_onnx.py --int8 models/sam3_image_encoder.onnx` writes. Its linear layers compute in int8. On an i9-12900K it encodes an image in 4.3 s against 7.5 s, and on an M5 CPU in 6.1 s against 7.1 s. It gives the same text detections, with scores within 0.007, and its tracked masks have a mean IoU of 0.994 with float32's, as PyTorch has 0.993 between bfloat16 and float32. The other models stay float32: with int8 tracker models, one of two tracked objects was lost, and int8 decoders ran no faster. `--int8` works with `--device cpu` and also in `infer_video_onnx.py` and `benchmark.py`.

## Benchmark

```sh
uv run benchmark.py --runtime torch
UV_PROJECT_ENVIRONMENT=.venv-gpu uv run --no-dev --no-group cpu --group gpu benchmark.py --device cuda
```

`benchmark.py` runs text, box, and point prompts on `images/bus.jpg`, then point, box, and text prompts through 40 frames of `bedroom.mp4`. It prints the mean time of each model and of the whole in milliseconds, per image or per tracked frame. Each case runs once untimed first. A frame's time runs from the JPEG files of the video to the masks of its last frame, over the frame count, so it holds loading the frames in both runtimes. Compare the totals across runtimes: a model's time on the PyTorch side is that of the nearest module, which does less than the ONNX model. `--runtime torch` runs upstream as it ships, in bfloat16 on CUDA. `--device` takes the same values as `infer_onnx.py`. On WebGPU, a run returns before the GPU finishes, so `benchmark.py` reads back the smallest output of each run that keeps its outputs on the GPU, which adds about 5 ms to it.
