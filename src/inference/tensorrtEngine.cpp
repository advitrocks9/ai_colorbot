// TensorRT engine loading and ONNX-to-engine build pipeline.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "inference/tensorrtEngine.h"
#include "app.h"

TrtLogger gLogger;

void TrtLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    if (severity > nvinfer1::ILogger::Severity::kWARNING) return;

    std::string devMsg = msg;
    // Translate incompatible engine error into a clear user action
    if (devMsg.find("magicTag") != std::string::npos ||
        devMsg.find("old deserialization") != std::string::npos)
    {
        std::cout << "[TensorRT] ERROR: Engine is incompatible with this TensorRT version. "
                     "Delete the .engine file and provide the .onnx - "
                     "the engine will be rebuilt automatically.\n";
    }
    else
    {
        std::cout << "[TensorRT] " << severityLevelName(severity) << ": " << msg << std::endl;
    }
}

const char* TrtLogger::severityLevelName(nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
    case nvinfer1::ILogger::Severity::kERROR:          return "ERROR";
    case nvinfer1::ILogger::Severity::kWARNING:        return "WARNING";
    case nvinfer1::ILogger::Severity::kINFO:           return "INFO";
    case nvinfer1::ILogger::Severity::kVERBOSE:        return "VERBOSE";
    default:                                           return "UNKNOWN";
    }
}

nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFile,
                                          nvinfer1::IRuntime* runtime)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "[TensorRT] Cannot open engine file: " << engineFile << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), static_cast<std::streamsize>(size));
    file.close();

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine)
    {
        std::cerr << "[TensorRT] Engine deserialization failed: " << engineFile << std::endl;
        return nullptr;
    }

    if (config.verbose)
        std::cout << "[TensorRT] Engine loaded: " << engineFile << std::endl;

    return engine;
}

nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile,
                                           nvinfer1::ILogger& logger)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvinfer1::IBuilderConfig*     cfg     = builder->createBuilderConfig();

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxFile.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "[TensorRT] ONNX parse failed: " << onnxFile << std::endl;
        delete parser; delete network; delete builder; delete cfg;
        return nullptr;
    }

    // Set dynamic shape profile for flexible input resolution (160→640)
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, 3, 160, 160));
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(1, 3, 320, 320));
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(1, 3, 640, 640));
    cfg->addOptimizationProfile(profile);

    // Apply precision from config (8 = FP8, 16 = FP16, default FP16)
    if (config.export_precision == 16)
    {
        if (config.verbose) std::cout << "[TensorRT] Precision: FP16" << std::endl;
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if (config.export_precision == 8)
    {
        if (config.verbose) std::cout << "[TensorRT] Precision: FP8" << std::endl;
        cfg->setFlag(nvinfer1::BuilderFlag::kFP8);
    }
    else
    {
        std::cerr << "[TensorRT] Unknown precision " << config.export_precision
                  << ", defaulting to FP16." << std::endl;
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    std::cout << "[TensorRT] Building engine (this may take several minutes)..." << std::endl;

    auto plan = builder->buildSerializedNetwork(*network, *cfg);
    if (!plan)
    {
        std::cerr << "[TensorRT] Engine build failed" << std::endl;
        delete parser; delete network; delete builder; delete cfg;
        return nullptr;
    }

    nvinfer1::IRuntime*    runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine  = runtime->deserializeCudaEngine(plan->data(),
                                                                     plan->size());

    // Save engine to disk
    std::string engineFile = onnxFile.substr(0, onnxFile.find_last_of('.')) + ".engine";
    std::ofstream ef(engineFile, std::ios::binary);
    if (ef)
    {
        ef.write(static_cast<const char*>(plan->data()),
                 static_cast<std::streamsize>(plan->size()));
        ef.close();
        std::cout << "[TensorRT] Engine saved: " << engineFile << std::endl;
    }
    else
    {
        std::cerr << "[TensorRT] Could not write engine file: " << engineFile << std::endl;
    }

    delete plan; delete runtime; delete parser; delete network; delete cfg; delete builder;
    return engine;
}
