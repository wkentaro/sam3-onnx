import pathlib
import tempfile

import numpy as np
import onnxruntime
import torch

from litetext import LiteTextLanguageEncoder


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def forward_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    def forward(
        self, embeds: torch.Tensor, return_all_tokens: bool, input_is_embeddings: bool
    ) -> torch.Tensor:
        assert return_all_tokens and input_is_embeddings
        # Mixing positions makes padding before encoding change valid features.
        return embeds + embeds.mean(dim=1, keepdim=True)


def test_litetext_padding() -> None:
    torch.manual_seed(0)
    model = LiteTextLanguageEncoder(_Encoder(), torch.nn.Linear(4, 256)).eval()
    tokens = torch.tensor([[1, 2, 3] + [0] * 13])
    with torch.inference_mode():
        embeds = model.encoder.forward_embedding(tokens)
        memory = model.projector(embeds + embeds.mean(dim=1, keepdim=True))
        mask, actual_memory, actual_embeds = model(tokens)
        assert mask.shape == (1, 32)
        assert actual_memory.shape == (32, 1, 256)
        assert actual_embeds.shape == (32, 1, 4)
        assert torch.equal(mask[:, :16], tokens == 0)
        assert mask[:, 16:].all()
        torch.testing.assert_close(actual_memory[:16], memory.transpose(0, 1))
        torch.testing.assert_close(actual_embeds[:16], embeds.transpose(0, 1))
        assert torch.count_nonzero(actual_memory[16:]) == 0
        assert torch.count_nonzero(actual_embeds[16:]) == 0

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "encoder.onnx"
        torch.onnx.export(
            model,
            (tokens,),
            path,
            input_names=["tokens"],
            output_names=["text_attention_mask", "text_memory", "text_embeds"],
            opset_version=21,
        )
        session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        for values in [tokens, torch.ones_like(tokens), torch.zeros_like(tokens)]:
            with torch.inference_mode():
                expected = model(values)
            actual = session.run(None, {"tokens": values.numpy()})
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e.numpy(), atol=1e-6)


if __name__ == "__main__":
    test_litetext_padding()
    print("ok")
