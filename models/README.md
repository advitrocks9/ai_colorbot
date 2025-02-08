# models/

Place TensorRT engine files (`.engine`) or ONNX models (`.onnx`) here.

The application will auto-detect all files in this directory at startup.
If a `.engine` file exists for a given model name, it takes priority over the `.onnx` version.

## Exporting a model

```bash
# Example: export YOLOv10n to FP16 TensorRT engine
trtexec --onnx=yolo10n.onnx --saveEngine=yolo10n.engine --fp16
```

## Tested models

| Model        | Postprocess setting |
|-------------|---------------------|
| YOLOv8n     | `yolo8`             |
| YOLOv10n    | `yolo10`            |
| YOLOv11n    | `yolo11`            |
| YOLOv12n    | `yolo12`            |
