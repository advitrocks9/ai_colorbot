// TensorRT utilities — logger, engine loading, and ONNX→engine export.
#ifndef TENSORRT_ENGINE_H
#define TENSORRT_ENGINE_H

#include "NvInfer.h"
#include <string>

// TrtLogger filters TensorRT log messages and translates incompatible-engine
// errors into a human-readable action message.
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
    static const char* severityLevelName(Severity severity);
};

extern TrtLogger gLogger;

// Load a serialised TensorRT engine from a .engine file.
nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFile,
                                          nvinfer1::IRuntime* runtime);

// Build a TensorRT engine from an ONNX model, applying precision flags from config.
// Saves the resulting engine to a .engine file alongside the .onnx source.
nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile,
                                           nvinfer1::ILogger& logger);

#endif // TENSORRT_ENGINE_H
