// TensorRT logger, engine loading, and ONNX-to-engine export.
#ifndef TENSORRT_ENGINE_H
#define TENSORRT_ENGINE_H

#include "NvInfer.h"
#include <string>

// Translates incompatible-engine errors into actionable messages.
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
    static const char* severityLevelName(Severity severity);
};

extern TrtLogger gLogger;

nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFile,
                                          nvinfer1::IRuntime* runtime);

nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile,
                                           nvinfer1::ILogger& logger);

#endif // TENSORRT_ENGINE_H
