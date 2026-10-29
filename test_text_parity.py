"""CPU integration check against Sam3Processor, using pre-exported models."""

import argparse
import pathlib
from unittest.mock import patch

import numpy as np
import onnxruntime
import PIL.Image
import torch
from osam._models.yoloworld.clip import tokenize
from sam3.model.decoder import TransformerDecoder  # type: ignore[unresolved-import]
from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[unresolved-import]
from sam3.model_builder import _create_position_encoding  # type: ignore[unresolved-import]
from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]
from torchvision.transforms import v2

from export_onnx import _ImageEncoder
from infer_onnx import preprocess_image


def test_preprocess_image() -> None:
    processor = Sam3Processor(model=None, device="cpu")
    encoder = _ImageEncoder(processor)
    rng = np.random.default_rng(0)
    for height, width in [(23, 37), (1201, 1503), (1008, 1008)]:
        image = PIL.Image.fromarray(rng.integers(0, 256, (height, width, 3), np.uint8))
        actual = encoder._transform(torch.from_numpy(preprocess_image(image).copy()))
        expected = processor.transform(v2.functional.to_image(image))
        # PIL's two-pass uint8 resize can round one pixel level differently.
        torch.testing.assert_close(actual, expected, rtol=0, atol=2 / 255 + 1e-6)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apple-image", type=pathlib.Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(8)
    options = onnxruntime.SessionOptions()  # type: ignore[possibly-missing-attribute]
    options.intra_op_num_threads = 8
    sessions = [
        onnxruntime.InferenceSession(
            f"models/sam3_{name}.onnx", options, providers=["CPUExecutionProvider"]
        )
        for name in ("image_encoder", "language_encoder", "decoder")
    ]
    image_session, language_session, decoder_session = sessions

    # SAM3 initializes two positional caches on CUDA even for a CPU model.
    get_coords = TransformerDecoder._get_coords
    with (
        patch(
            "sam3.model_builder._create_position_encoding",
            side_effect=lambda precompute_resolution=None: _create_position_encoding(),
        ),
        patch.object(
            TransformerDecoder,
            "_get_coords",
            side_effect=lambda H, W, device: get_coords(H, W, "cpu"),
        ),
    ):
        model = build_sam3_image_model(device="cpu")
    processor = Sam3Processor(model, device="cpu")
    encoder = _ImageEncoder(processor)

    checks = []
    for path, prompt in [(args.apple_image, "apple"), ("images/bus.jpg", "person")]:
        image = PIL.Image.open(path).convert("RGB")
        pixels = preprocess_image(image)
        preprocessed = encoder._transform(torch.from_numpy(pixels.copy()))
        reference_image = processor.transform(v2.functional.to_image(image))
        errors = {"preprocessing": float((preprocessed - reference_image).abs().max())}

        reference = processor.set_text_prompt(
            prompt=prompt, state=processor.set_image(image)
        )
        backbone = reference["backbone_out"]
        image_names = [f"backbone_fpn_{i}" for i in range(3)] + ["vision_pos_enc_2"]
        expected_image = [
            *(t.numpy() for t in backbone["backbone_fpn"]),
            backbone["vision_pos_enc"][2].numpy(),
        ]
        image_outputs = image_session.run(image_names, {"image": pixels})
        errors["image encoder"] = max(
            float(np.abs(a - b).max())
            for a, b in zip(image_outputs, expected_image, strict=True)
        )

        language_mask, language_features, _ = language_session.run(
            None, {"tokens": tokenize(texts=[prompt], context_length=32)}
        )
        np.testing.assert_array_equal(language_mask, backbone["language_mask"].numpy())
        errors["language encoder"] = float(
            np.abs(language_features - backbone["language_features"].numpy()).max()
        )

        (box_input,) = [
            i for i in decoder_session.get_inputs() if i.name == "box_coords"
        ]
        n = box_input.shape[0]
        boxes = {
            "box_coords": np.zeros((n, 1, 4), dtype=np.float32),
            "box_labels": np.ones((n, 1), dtype=np.int64),
            "box_masks": np.ones((1, n), dtype=np.bool_),
        }
        scores = decoder_session.run(
            ["scores"],
            {
                **dict(zip(image_names, image_outputs, strict=True)),
                "language_mask": language_mask,
                "language_features": language_features,
                **boxes,
            },
        )[0]
        shared_scores = decoder_session.run(
            ["scores"],
            {
                **dict(zip(image_names, expected_image, strict=True)),
                "language_mask": backbone["language_mask"].numpy(),
                "language_features": backbone["language_features"].numpy(),
                **boxes,
            },
        )[0]
        assert isinstance(scores, np.ndarray)
        assert isinstance(shared_scores, np.ndarray)
        expected_scores = reference["scores"].numpy()
        print(f"{path}, prompt={prompt!r}", flush=True)
        for stage, error in errors.items():
            print(f"  {stage}: max absolute error={error:.8f}", flush=True)
        first = next((s for s, e in errors.items() if e > 1e-4), "none")
        print(f"  first divergence over 1e-4: {first}", flush=True)
        for label, actual in [
            ("pipeline", scores),
            ("decoder, shared inputs", shared_scores),
        ]:
            error = (
                float(np.abs(np.sort(actual) - np.sort(expected_scores)).max(initial=0))
                if actual.shape == expected_scores.shape
                else float("inf")
            )
            print(
                f"  {label}: count ONNX/PyTorch={len(actual)}/{len(expected_scores)}, "
                f"max sorted score error={error:.8f}",
                flush=True,
            )
            checks.append((actual, expected_scores))

    for actual, expected in checks:
        np.testing.assert_allclose(
            np.sort(actual), np.sort(expected), rtol=0, atol=0.01
        )


if __name__ == "__main__":
    main()
    test_preprocess_image()
