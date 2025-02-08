# Architecture

## Thread model

| Thread | Source | Role |
|--------|--------|------|
| `captureThread` | `src/capture/capture.cpp` | DXGI frame capture, circle-mask, FPS limiter |
| `Detector::inferenceThread` | `src/detection/detector.cpp` | TensorRT preprocess → infer → postprocess |
| `mouseThreadFunction` | `src/main.cpp` | PID target selection, Makcu serial write |
| `overlayThread` | `src/overlay/overlay.cpp` | Win32 + OpenGL + ImGui settings panel |
| `displayThread` | `src/overlay/visuals.cpp` | OpenCV debug window with bounding boxes |
| `udpListenerThread` | `src/main.cpp` | Remote aimbot/triggerbot enable/disable |

## Data flow

```
DXGI Desktop Duplication (D3D11 texture)
    │ GPU zero-copy (CUDA D3D11 interop)  ─── or ───  CPU memcpy
    ▼
frameMutex / latestFrameGpu (cv::cuda::GpuMat)
    │
    ├─► displayThread  →  cv::imshow("Debug")
    │
    └─► Detector::inferenceThread
            │  cuDNN + TensorRT FP16
            ▼
        detectedBoxes / detectedClasses (detector.detectionCV)
            │
            └─► mouseThreadFunction
                    │  sortTargets() → AimbotTarget*
                    │  MouseThread::moveMousePivot()
                    │  PID → Makcu serial (USB HID emulator)
                    └─► pressMouse() / releaseMouse()
```

## Config propagation

The overlay thread holds `configMutex` while drawing ImGui widgets.
Dirty-flag atomics (`detection_resolution_changed`, `detector_model_changed`,
`show_window_changed`, …) are set in-frame when a value changes, and consumed
by the relevant worker threads on their next iteration.

## Capture pipeline (CUDA path)

```
D3D11Texture2D (GPU)
  │ cuGraphicsD3D11RegisterResource
  │ cuGraphicsMapResources
  ▼
CUDA surface → cv::cuda::GpuMat (BGRA, detection_resolution × detection_resolution)
  │ optional: cv::cuda::circle mask
  ▼
Shared frame buffer (latestFrameGpu)
```

## Inference pipeline

```
cv::cuda::GpuMat (BGR, NxN)
  │ cv::cuda::resize → 1×3×H×W letterbox
  │ normalize [0,1] on GPU
  ▼
TensorRT FP16 engine (binding: "images" → "output0")
  │ optional: CUDA graph replay
  ▼
Raw output tensor (device memory)
  │ postProcess() — YOLO decoder variant
  │ NMS (CPU, OpenCV)
  ▼
std::vector<cv::Rect> boxes, std::vector<int> classes
```
