#!/usr/bin/env python3

import argparse
import functools
import pathlib
import sys
import typing

import cv2
import imgviz
import numpy as np
import onnx
import onnxruntime
import PIL.Image
from loguru import logger
from numpy.typing import NDArray
from onnx import numpy_helper
from osam._models.yoloworld.clip import tokenize
from osam.types._model import _setup_cuda
from osam.types._model import _webgpu_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=pathlib.Path,
        help="Path to the input image.",
        required=True,
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        help="Text prompt for segmentation.",
    )
    parser.add_argument(
        "--box-prompt",
        type=str,
        action="append",
        nargs="?",
        const="0,0,0,0",
        help="Box prompt of an example object in format: cx,cy,w,h[,label] "
        "(normalized, label 1 for positive and 0 for negative). Repeatable, and "
        "combinable with --text-prompt.",
    )
    parser.add_argument(
        "--point-prompt",
        type=str,
        action="append",
        help="Point prompt for segmenting one object in format: x,y[,label] "
        "(normalized, label 1 for foreground and 0 for background). Repeatable.",
    )
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="Predict one mask for --point-prompt instead of picking the best of "
        "3, which suits several points.",
    )
    parser.add_argument(
        "--litetext",
        choices=["s0", "s1", "l"],
        help="Use a SAM3-LiteText language encoder exported with --litetext.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help=DEVICE_HELP,
    )
    parser.add_argument("--fp16", action="store_true", help=FP16_HELP)
    args = parser.parse_args()
    logger.debug("input: {}", args.__dict__)

    if args.point_prompt and (args.text_prompt or args.box_prompt):
        parser.error("--point-prompt can't be combined with other prompts")
    if args.single_mask and not args.point_prompt:
        parser.error("--single-mask requires --point-prompt")
    if not (args.text_prompt or args.box_prompt or args.point_prompt):
        parser.error("one of --text-prompt, --box-prompt, --point-prompt is required")

    if args.box_prompt and len(args.box_prompt) > 1 and "0,0,0,0" in args.box_prompt:
        parser.error("--box-prompt without a value can't be repeated")

    if args.box_prompt:
        boxes = [[float(x) for x in b.split(",")] for b in args.box_prompt]
        if any(len(b) not in (4, 5) for b in boxes):
            logger.error("box_prompt must have 4 or 5 values: cx, cy, w, h[, label]")
            sys.exit(1)
        if any(len(b) == 5 and b[4] not in (0, 1) for b in boxes):
            logger.error("box_prompt label must be 0 or 1")
            sys.exit(1)
        args.box_prompt = [b if len(b) == 5 else [*b, 1] for b in boxes]

    if args.point_prompt:
        points = [[float(x) for x in p.split(",")] for p in args.point_prompt]
        if any(len(p) not in (2, 3) for p in points):
            logger.error("point_prompt must have 2 or 3 values: x, y[, label]")
            sys.exit(1)
        args.point_prompt = [p if len(p) == 3 else [*p, 1] for p in points]

    if args.box_prompt == [[0, 0, 0, 0, 1]]:
        image: NDArray[np.uint8] = imgviz.asrgb(imgviz.io.imread(args.image))

        logger.info("please select box prompt in the image window")
        x, y, w, h = cv2.selectROI(
            "Select box prompt and press ENTER or SPACE",
            image[:, :, ::-1],
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()
        if [x, y, w, h] == [0, 0, 0, 0]:
            logger.warning("no box prompt selected, exiting")
            sys.exit(1)

        box = [
            (x + w / 2) / image.shape[1],
            (y + h / 2) / image.shape[0],
            w / image.shape[1],
            h / image.shape[0],
        ]
        logger.debug("box_prompt: {!r}", ",".join(f"{x:.3f}" for x in box))
        args.box_prompt = [[*box, 1]]

    return args


def preprocess_image(image: PIL.Image.Image) -> NDArray[np.uint8]:
    # Sam3Processor uses antialiased bilinear resizing, not PIL's default bicubic.
    return np.asarray(
        image.resize((1008, 1008), resample=PIL.Image.Resampling.BILINEAR)
    ).transpose(2, 0, 1)


def postprocess_decoder_output(
    boxes: NDArray,
    masks: NDArray,
    image_width: int,
    image_height: int,
) -> tuple[NDArray, NDArray]:
    boxes = boxes * np.array(
        [image_width, image_height, image_width, image_height], dtype=np.float32
    )
    if len(masks) == 0:
        return boxes, np.empty((0, image_height, image_width), dtype=bool)
    masks = np.array(
        [
            cv2.resize(
                mask[0],
                dsize=(image_width, image_height),
                interpolation=cv2.INTER_LINEAR,
            )
            > 0.5
            for mask in masks
        ]
    )
    return boxes, masks


def postprocess_point_decoder_output(
    iou_predictions: NDArray,
    low_res_masks: NDArray,
    image_width: int,
    image_height: int,
) -> tuple[NDArray, float]:
    best = iou_predictions[0].argmax()
    mask_logits: NDArray[np.float32] = low_res_masks[0, best]
    # Fill background holes up to 256 px, as SAM2Transforms.postprocess_masks.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_logits <= 0).astype(np.uint8), connectivity=8
    )
    is_hole = (labels > 0) & (stats[labels, cv2.CC_STAT_AREA] <= 256)
    mask_logits = np.where(is_hole, np.float32(10), mask_logits)
    mask = (
        cv2.resize(
            mask_logits,
            dsize=(image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        > 0
    )
    return mask, float(iou_predictions[0, best])


def run_point_decoder(
    sess_point_decode: onnxruntime.InferenceSession,
    sam2_features: list[typing.Any],
    point_prompt: list[list[float]],
    multimask_output: bool = True,
) -> tuple[NDArray, NDArray]:
    """Return 3 masks with multimask_output, or else 1 as SAM3 picks it."""
    output_names = (
        ["iou_predictions", "low_res_masks"]
        if multimask_output
        else ["single_iou_predictions", "single_low_res_masks"]
    )
    points = np.array(point_prompt, dtype=np.float32)
    inputs: dict[str, NDArray] = {
        "image_embeddings": sam2_features[0],
        "high_res_features1": sam2_features[1],
        "high_res_features2": sam2_features[2],
        "point_coords": points[None, :, :2] * 1008,
        "point_labels": points[None, :, 2],
        "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
    }
    iou_predictions, low_res_masks = typing.cast(
        list[NDArray], sess_point_decode.run(output_names, inputs)
    )
    # Refine with the best mask from the first pass as the mask prompt.
    inputs["mask_input"] = low_res_masks[:, [iou_predictions[0].argmax()]]
    inputs["has_mask_input"] = np.array([1], dtype=np.float32)
    iou_predictions, low_res_masks = typing.cast(
        list[NDArray], sess_point_decode.run(output_names, inputs)
    )
    return iou_predictions, low_res_masks


def get_max_boxes(session: onnxruntime.InferenceSession) -> int:
    (box_input,) = [i for i in session.get_inputs() if i.name == "box_coords"]
    return box_input.shape[0]


def make_box_inputs(
    session: onnxruntime.InferenceSession,
    boxes: typing.Sequence[typing.Sequence[float]],
) -> dict[str, NDArray]:
    """The decoder takes a fixed number of boxes, given as [cx, cy, w, h, label]
    rows, and masks out the unused ones."""
    max_boxes = get_max_boxes(session)
    box_coords = np.zeros((max_boxes, 1, 4), dtype=np.float32)
    box_labels = np.ones((max_boxes, 1), dtype=np.int64)
    box_masks = np.ones((1, max_boxes), dtype=np.bool_)
    for i, box in enumerate(boxes):
        box_coords[i, 0] = box[:4]
        box_labels[i, 0] = box[4]
        box_masks[0, i] = False
    return {"box_coords": box_coords, "box_labels": box_labels, "box_masks": box_masks}


def load_for_coreml(onnx_file: str) -> bytes:
    """Rewrite Gemm(transB=0) to Gemm(transB=1) with W {N, K}.

    The CoreML EP loads transB=0 in minutes, as it writes the transposed W into
    the model as text.
    """
    model: onnx.ModelProto = onnx.load(onnx_file)
    initializers = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue
        trans_b = next((a for a in node.attribute if a.name == "transB"), None)
        if trans_b is not None and trans_b.i == 1:
            continue
        weight = initializers[node.input[1]]
        weight.CopyFrom(
            numpy_helper.from_array(numpy_helper.to_array(weight).T, weight.name)
        )
        if trans_b is None:
            node.attribute.append(onnx.helper.make_attribute("transB", 1))
        else:
            trans_b.i = 1
    return model.SerializeToString()


# The image encoder outputs that each decoder reads. Fetching these alone
# skips vision_pos_enc_0 and vision_pos_enc_1, 106 MB of constants that no
# model reads and that would be copied from the GPU on every run.
DECODER_FEATURES: list[str] = [
    "vision_pos_enc_2",
    "backbone_fpn_0",
    "backbone_fpn_1",
    "backbone_fpn_2",
]
POINT_DECODER_FEATURES: list[str] = [
    "image_embeddings",
    "high_res_features1",
    "high_res_features2",
]

DEVICES = ["cpu", "cuda", "coreml", "webgpu"]
DEVICE_HELP = (
    "Device to run the models on. cuda needs the gpu dependency group, coreml "
    "needs macOS, and webgpu needs the webgpu dependency group. coreml runs the "
    "image encoder, and the other models on WebGPU with the webgpu group "
    "installed, or else on the CPU."
)


FP16_HELP = (
    "Run the float16 models of postprocess_onnx.py --fp16 on the GPU, which is "
    "up to twice as fast. Models that --device leaves on the CPU stay float32."
)


def model_device(name: str, device: str, has_webgpu: bool) -> str:
    """The device that runs one model.

    With coreml, CoreML runs the image encoder alone: it runs it 3 times as
    fast as WebGPU does on an Apple GPU, but it can't compile the tracker's
    memory attention, whose input grows as frames fill the memory, and the
    decoders take longer to compile than it saves them. WebGPU runs those
    when its plugin is installed, and the CPU otherwise.
    """
    if device != "coreml" or name == "image_encoder":
        return device
    return "webgpu" if has_webgpu else "cpu"


def model_file(name: str, device: str, fp16: bool) -> str:
    """The float16 copy for a model that runs on a GPU, where it is faster.
    On a CPU it is slower than float32, which has more kernels."""
    return f"models/sam3_{name}{'_fp16' if fp16 and device != 'cpu' else ''}.onnx"


def _has_webgpu() -> bool:
    try:
        return _webgpu_device() is not None
    except RuntimeError:  # the plugin is not installed
        return False


@functools.cache
def _session_options(
    device: str,
) -> tuple[typing.Any, list[str | tuple[str, dict[str, str]]]]:
    """The session options and providers that run a model on a device."""
    providers: list[str | tuple[str, dict[str, str]]] = ["CPUExecutionProvider"]
    sess_options = onnxruntime.SessionOptions()  # type: ignore[possibly-missing-attribute]
    if device == "cuda":
        # Loads the CUDA libraries of osam[gpu] and shares one CUDA arena across
        # sessions, without which the three sessions OOM on a 12 GB GPU.
        if not _setup_cuda():
            logger.error(
                "CUDA is unavailable, run with: UV_PROJECT_ENVIRONMENT=.venv-gpu "
                "uv run --no-dev --no-group cpu --group gpu infer_onnx.py ..."
            )
            sys.exit(1)
        providers.insert(0, "CUDAExecutionProvider")
        sess_options.add_session_config_entry("session.use_env_allocators", "1")
    elif device == "coreml":
        available = onnxruntime.get_available_providers()  # type: ignore[possibly-missing-attribute]
        if "CoreMLExecutionProvider" not in available:
            logger.error("CoreML is unavailable, it needs onnxruntime on macOS")
            sys.exit(1)
        # onnxruntime's CoreML EP can't read external data.
        if pathlib.Path("models/sam3_image_encoder.onnx.data").exists():
            logger.error(
                "CoreML needs models with inline weights, run: "
                "uv run postprocess_onnx.py models/*.onnx && rm models/*.onnx.data"
            )
            sys.exit(1)
        # MLProgram supports more operators than the default NeuralNetwork, and
        # leaving out the Neural Engine keeps the GPU's accuracy and speed.
        # ponytail: no ModelCacheDirectory, which saves 3 s of the compile but
        # stores another 3.5 GB keyed by model path; add it for long sessions.
        coreml_options = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}
        providers.insert(0, ("CoreMLExecutionProvider", coreml_options))
    elif device == "webgpu":
        # osam's onnxruntime-ep-webgpu plugin, on the adapter osam picks.
        try:
            webgpu_device = _webgpu_device()
        except RuntimeError:  # the plugin is not installed
            logger.error(
                "WebGPU is unavailable, run with: "
                "uv run --no-dev --group webgpu infer_onnx.py ..."
            )
            sys.exit(1)
        if webgpu_device is None:
            logger.error("WebGPU found no GPU adapter")
            sys.exit(1)
        # Unlike CoreML, it takes inputs whose shapes change, as the memory of
        # the tracker does.
        sess_options.add_provider_for_devices([webgpu_device], {})
        providers.insert(0, "WebGpuExecutionProvider")
    return sess_options, providers


def create_sessions(
    device: str, names: list[str], fp16: bool = False
) -> list[onnxruntime.InferenceSession]:
    if fp16 and device == "cpu":
        logger.error("--fp16 needs a GPU: --device cuda, coreml, or webgpu")
        sys.exit(1)
    has_webgpu = device == "coreml" and _has_webgpu()
    sessions: list[onnxruntime.InferenceSession] = []
    for name in names:
        device_of_model = model_device(name, device, has_webgpu)
        sess_options, providers = _session_options(device_of_model)
        model: str | bytes = model_file(name, device_of_model, fp16)
        if not pathlib.Path(model).exists():
            hint = (
                "postprocess_onnx.py --fp16 models/*.onnx" if fp16 else "export_onnx.py"
            )
            logger.error("{} is missing, create it with: uv run {}", model, hint)
            sys.exit(1)
        if device_of_model == "coreml":
            # ponytail: rewritten on every run (~2 s); cache it if that matters.
            model = load_for_coreml(model)
        session = onnxruntime.InferenceSession(
            model,
            sess_options,
            # A plugin provider comes from sess_options, by device.
            providers=None if device_of_model == "webgpu" else providers,
        )
        # onnxruntime silently falls back to CPU for a provider it can't use.
        (provider, *_) = providers
        provider = provider if isinstance(provider, str) else provider[0]
        if provider not in session.get_providers():
            logger.error("{} is not running on {}", name, provider)
            sys.exit(1)
        logger.debug("{} runs on {}", name, provider)
        sessions.append(session)
    return sessions


# The device of the outputs that run_on_device keeps, by provider.
_OUTPUT_DEVICES: dict[str, str] = {
    "CUDAExecutionProvider": "cuda",
    "WebGpuExecutionProvider": "webgpu",
}


def run_on_device(
    session: onnxruntime.InferenceSession,
    output_names: list[str],
    inputs: dict[str, typing.Any],
) -> list[typing.Any]:
    """session.run, but on CUDA and WebGPU the outputs stay on the GPU as
    OrtValues.

    Another session on the same device takes them as inputs, as session.run
    takes OrtValues, without the copy to the CPU and back. Elsewhere it
    returns numpy arrays as session.run does.
    """
    device = _OUTPUT_DEVICES.get(session.get_providers()[0])
    if device is None:
        return list(session.run(output_names, inputs))
    binding = session.io_binding()
    for name, value in inputs.items():
        if isinstance(value, onnxruntime.OrtValue):  # type: ignore[possibly-missing-attribute]
            binding.bind_ortvalue_input(name, value)
        else:
            binding.bind_cpu_input(name, value)
    for name in output_names:
        binding.bind_output(name, device)
    session.run_with_iobinding(binding)
    return binding.get_outputs()


def main():
    args = parse_args()

    decoder_names: list[str] = (
        ["point_decoder"] if args.point_prompt else ["language_encoder", "decoder"]
    )
    if args.litetext and not args.point_prompt:
        decoder_names[0] = f"litetext_{args.litetext}_language_encoder"
    sess_image, *sess_decoders = create_sessions(
        device=args.device, names=["image_encoder", *decoder_names], fp16=args.fp16
    )
    logger.debug("providers: {}", sess_image.get_providers())

    image: PIL.Image.Image = PIL.Image.open(args.image).convert("RGB")
    logger.debug("original image size: {}", image.size)

    logger.debug("running image encoder...")
    output_names: list[str] = (
        POINT_DECODER_FEATURES if args.point_prompt else DECODER_FEATURES
    )
    features = run_on_device(
        sess_image, output_names, {"image": preprocess_image(image)}
    )
    logger.debug("finished running image encoder")

    if args.point_prompt:
        if args.single_mask and "single_low_res_masks" not in [
            o.name for o in sess_decoders[0].get_outputs()
        ]:
            logger.error(
                "models/sam3_point_decoder.onnx predates --single-mask, "
                "re-export it with: uv run export_onnx.py"
            )
            sys.exit(1)
        logger.debug("running point decoder...")
        iou_predictions, low_res_masks = run_point_decoder(
            sess_point_decode=sess_decoders[0],
            sam2_features=features,
            point_prompt=args.point_prompt,
            multimask_output=not args.single_mask,
        )
        mask, score = postprocess_point_decoder_output(
            iou_predictions=iou_predictions,
            low_res_masks=low_res_masks,
            image_width=image.width,
            image_height=image.height,
        )
        logger.debug("output: {}", {"mask": mask.shape, "score": score})
        viz = imgviz.instances2rgb(
            image=np.asarray(image),
            masks=mask[None],
            labels=[1],
            captions=[f"point: {score:.0%}"],
            font_size=max(1, min(image.size) // 40),
        )
        imgviz.io.imshow(viz)
        return

    sess_language, sess_decode = sess_decoders

    logger.debug("running language encoder...")
    text_prompt: str = args.text_prompt if args.text_prompt else "visual"
    output = sess_language.run(
        None,
        {
            "tokens": tokenize(
                texts=[text_prompt],
                context_length=16 if args.litetext else 32,
                truncate=bool(args.litetext),
            )
        },
    )
    assert len(output) == 3
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    language_mask: NDArray = output[0]
    language_features: NDArray = output[1]
    # language_embeds: NDArray = output[2]
    logger.debug("finished running language encoder")

    logger.debug("running decoder...")
    box_prompt: list[list[float]] = args.box_prompt or []
    max_boxes = get_max_boxes(sess_decode)
    if len(box_prompt) > max_boxes:
        logger.error("box_prompt must have at most {} boxes", max_boxes)
        sys.exit(1)
    box_inputs = make_box_inputs(sess_decode, box_prompt)
    output = sess_decode.run(
        ["boxes", "scores", "masks"],
        {
            **dict(zip(DECODER_FEATURES, features)),
            "language_mask": language_mask,
            "language_features": language_features,
            **box_inputs,
        },
    )
    assert len(output) == 3
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    boxes: NDArray = output[0]
    scores: NDArray = output[1]
    masks: NDArray = output[2]
    logger.debug("finished running decoder")

    boxes, masks = postprocess_decoder_output(
        boxes=boxes,
        masks=masks,
        image_width=image.width,
        image_height=image.height,
    )

    logger.debug(
        "output: {}",
        {"masks": masks.shape, "boxes": boxes.shape, "scores": scores.shape},
    )

    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks,
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{text_prompt}: {s:.0%}" for s in scores],
        font_size=max(1, min(image.size) // 40),
    )
    imgviz.io.imshow(viz)


if __name__ == "__main__":
    main()
