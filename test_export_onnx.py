import sam3.model.vitdet  # type: ignore[unresolved-import]
import torch

from export_onnx import _rope_image_encoder


def test_rope_image_encoder() -> None:
    generator = torch.Generator().manual_seed(0)
    xq = torch.randn(2, 3, 5, 8, generator=generator)
    xk = torch.randn(2, 3, 5, 8, generator=generator)
    angles = torch.rand(5, 4, generator=generator) * 6
    args = (xq, xk, angles.cos(), angles.sin())

    expected = sam3.model.vitdet.apply_rotary_enc2(*args)
    for actual, wanted in zip(_rope_image_encoder(*args), expected):
        torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=0)


if __name__ == "__main__":
    test_rope_image_encoder()
    print("ok")
