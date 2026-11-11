# Video tracking API

`video_onnx.VideoInference` ports SAM3's video predictor (`Sam3VideoInferenceWithInstanceInteractivity`). A text prompt finds the objects on every frame with `sam3_decoder.onnx`, and a `Tracker` tracks each one. Points then refine an object or add one, and `remove_object` drops one.

`track_onnx.Tracker` ports SAM3's video tracker (`Sam3TrackerPredictor`) to numpy on top of four models: `sam3_point_decoder.onnx`, `sam3_memory_encoder.onnx`, `sam3_memory_attention.onnx`, and `sam3_mask_downsampler.onnx`. It takes point, box, and mask prompts for several objects, refinement clicks on any frame, forward and reverse propagation, and object removal, with the same method names as the PyTorch predictor.
