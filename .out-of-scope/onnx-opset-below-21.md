# ONNX opsets below 21

This project exports its models at opset 21 only. Exporting at a lower opset,
such as 16, is not supported.

## Why this is out of scope

Each opset is a separate export to test: the export has to succeed, and the
outputs have to match PyTorch for every model (image encoder, language
encoder, decoder, point decoder). The repo tests one opset so that this
check stays cheap. Current onnxruntime releases, including its CUDA and
CoreML execution providers, run opset 21 models.

Lowering the opset is not a one-line change. In #4, the image encoder export
at `opset_version=16` fails inside a convolution with "Cannot insert a Tensor
that requires grad as a constant", so supporting it would need changes beyond
the export call.

Converting the exported models down with `onnx.version_converter` doesn't
work either. Opset 16 has no `LayerNormalization`, which the models use, and
the converter fails on `ConstantOfShape` in the point decoder.

## Prior requests

- #4: "when export_onnx of opset=16, unwork."
