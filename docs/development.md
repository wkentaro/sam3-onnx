# Development

## Inference with PyTorch

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

## Convert an older export

Convert models from an older export in place, as `export_onnx.py` does:

```sh
uv run postprocess_onnx.py models/*.onnx && rm models/*.onnx.data
```

## Upload models to Hugging Face Hub

```sh
uvx hf upload wkentaro/sam3-onnx-models-v0.7.0 models/ --include '*.onnx'
```

## Check text-prompt scores against PyTorch on CPU

Save the apple image from [issue #2](https://github.com/wkentaro/sam3-onnx/issues/2), then run:

```sh
uv run test_text_parity.py --apple-image /path/to/apple.jpg
```

The check compares preprocessing, image and language features, and decoder scores for apples and `images/bus.jpg` with "person". It also feeds PyTorch features to the ONNX decoder to isolate decoder errors. Both pipelines must return the same detection count and sorted scores within 0.01. PIL bilinear resizing can differ from PyTorch's tensor resize by one uint8 level.

## Check video tracking against PyTorch on CUDA

```sh
uv run test_video_parity.py
uv run test_video_inference_parity.py
```

The check runs point, refinement, reverse, box, mask, two-object, and removal cases on the first 60 frames of `bedroom.mp4` with the PyTorch tracker in float32, and fails if a frame's mask IoU drops below 0.99.

`test_video_inference_parity.py` checks `VideoInference` the same way against `build_sam3_video_model()`. It runs text and box prompts, forward and reverse, point refinement, an added object, and removal. It fails if the object ids differ, a mask IoU drops below 0.99, or a score differs by more than 0.01.
