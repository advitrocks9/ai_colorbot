// Detector implementation — TensorRT lifecycle, preprocessing, and postprocessing.
// The inference loop runs on its own thread and accepts frames produced by the
// capture thread via a single-slot condition variable handoff.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>

#include <algorithm>
#include <cuda_fp16.h>
#include <atomic>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>

#include "detection/detector.h"
#include "inference/tensorrtEngine.h"
#include "app.h"
#include "util/otherTools.h"
#include "detection/postProcess.h"

extern std::atomic<bool> detectionPaused;
extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;

static bool errorLogged = false;

Detector::Detector()
    : frameReady(false),
      shouldExit(false),
      detectionVersion(0),
      inputBufferDevice(nullptr),
      img_scale(1.0f),
      numClasses(0),
      useCudaGraph(false),
      cudaGraphCaptured(false)
{
    cudaStreamCreate(&stream);
    cvStream            = cv::cuda::Stream();
    preprocessCvStream  = cv::cuda::Stream();
    postprocessCvStream = cv::cuda::Stream();
    cudaGraph           = nullptr;
    cudaGraphExec       = nullptr;
    useCudaGraph        = config.use_cuda_graph;
}

Detector::~Detector()
{
    cudaStreamDestroy(stream);
    destroyCudaGraph();

    for (auto& buf : pinnedOutputBuffers)
        if (buf.second) cudaFreeHost(buf.second);

    for (auto& b : inputBindings)
        if (b.second) cudaFree(b.second);

    for (auto& b : outputBindings)
        if (b.second) cudaFree(b.second);

    if (inputBufferDevice) cudaFree(inputBufferDevice);
}

void Detector::destroyCudaGraph()
{
    if (cudaGraphCaptured)
    {
        cudaGraphExecDestroy(cudaGraphExec);
        cudaGraphDestroy(cudaGraph);
        cudaGraphCaptured = false;
    }
}

// Records one inference execution into a CUDA graph for subsequent low-overhead replay.
void Detector::captureCudaGraph()
{
    if (!useCudaGraph || cudaGraphCaptured) return;

    cudaStreamSynchronize(stream);

    cudaError_t st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] BeginCapture failed: " << cudaGetErrorString(st) << std::endl;
        return;
    }

    context->enqueueV3(stream);

    for (const auto& name : outputNames)
        if (pinnedOutputBuffers.count(name))
            cudaMemcpyAsync(pinnedOutputBuffers[name],
                            outputBindings[name],
                            outputSizes[name],
                            cudaMemcpyDeviceToHost,
                            stream);

    st = cudaStreamEndCapture(stream, &cudaGraph);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] EndCapture failed: " << cudaGetErrorString(st) << std::endl;
        return;
    }

    st = cudaGraphInstantiate(&cudaGraphExec, cudaGraph, 0);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] GraphInstantiate failed: " << cudaGetErrorString(st) << std::endl;
        return;
    }

    cudaGraphCaptured = true;
}

inline void Detector::launchCudaGraph()
{
    cudaError_t err = cudaGraphLaunch(cudaGraphExec, stream);
    if (err != cudaSuccess)
        std::cerr << "[Detector] GraphLaunch failed: " << cudaGetErrorString(err) << std::endl;
}

void Detector::getInputNames()
{
    inputNames.clear();
    inputSizes.clear();
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputNames.emplace_back(name);
            if (config.verbose)
                std::cout << "[Detector] Input tensor: " << name << std::endl;
        }
    }
}

void Detector::getOutputNames()
{
    outputNames.clear();
    outputSizes.clear();
    outputTypes.clear();
    outputShapes.clear();
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputNames.emplace_back(name);
            outputTypes[name] = engine->getTensorDataType(name);
            if (config.verbose)
                std::cout << "[Detector] Output tensor: " << name << std::endl;
        }
    }
}

void Detector::getBindings()
{
    for (auto& b : inputBindings)  if (b.second) cudaFree(b.second);
    for (auto& b : outputBindings) if (b.second) cudaFree(b.second);
    inputBindings.clear();
    outputBindings.clear();

    for (const auto& name : inputNames)
    {
        size_t size = inputSizes[name];
        if (size > 0) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess)
                inputBindings[name] = ptr;
            else
                std::cerr << "[Detector] Failed to allocate input for " << name
                          << ": " << cudaGetErrorString(err) << std::endl;
        }
    }

    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        if (size > 0) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess)
                outputBindings[name] = ptr;
            else
                std::cerr << "[Detector] Failed to allocate output for " << name
                          << ": " << cudaGetErrorString(err) << std::endl;
        }
    }
}

void Detector::initialize(const std::string& modelFile)
{
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    loadEngine(modelFile);

    if (!engine) {
        std::cerr << "[Detector] Engine loading failed" << std::endl;
        return;
    }

    context.reset(engine->createExecutionContext());
    if (!context) {
        std::cerr << "[Detector] Context creation failed" << std::endl;
        return;
    }

    getInputNames();
    getOutputNames();

    if (inputNames.empty()) {
        std::cerr << "[Detector] No input tensors found" << std::endl;
        return;
    }

    inputName = inputNames[0];
    context->setInputShape(inputName.c_str(), nvinfer1::Dims4{ 1, 3, 640, 640 });
    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "[Detector] Failed to set input dimensions" << std::endl;
        return;
    }

    for (const auto& inName : inputNames)
    {
        auto dims = context->getTensorShape(inName.c_str());
        auto dt   = engine->getTensorDataType(inName.c_str());
        inputSizes[inName] = getSizeByDim(dims) * getElementSize(dt);
    }

    for (const auto& outName : outputNames)
    {
        auto dims = context->getTensorShape(outName.c_str());
        auto dt   = engine->getTensorDataType(outName.c_str());
        outputSizes[outName] = getSizeByDim(dims) * getElementSize(dt);

        std::vector<int64_t> shape;
        for (int i = 0; i < dims.nbDims; ++i) shape.push_back(dims.d[i]);
        outputShapes[outName] = shape;
    }

    getBindings();

    if (!outputNames.empty())
    {
        const auto& mainOut = outputNames[0];
        auto outDims = context->getTensorShape(mainOut.c_str());
        numClasses = (config.postprocess == "yolo10")
            ? 11
            : static_cast<int>(outDims.d[1] - 4);
    }

    img_scale = static_cast<float>(config.detection_resolution) / 640.0f;

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    inputDims = dims;

    int c = static_cast<int>(dims.d[1]);
    int h = static_cast<int>(dims.d[2]);
    int w = static_cast<int>(dims.d[3]);

    resizedBuffer.create(h, w, CV_8UC3);
    floatBuffer.create(h, w, CV_32FC3);
    channelBuffers.resize(c);
    for (int i = 0; i < c; ++i)
        channelBuffers[i].create(h, w, CV_32F);

    for (const auto& name : inputNames)
        context->setTensorAddress(name.c_str(), inputBindings[name]);
    for (const auto& name : outputNames)
        context->setTensorAddress(name.c_str(), outputBindings[name]);

    if (config.use_pinned_memory)
    {
        for (const auto& outName : outputNames)
        {
            size_t size = outputSizes[outName];
            void* hostBuf = nullptr;
            if (cudaMallocHost(&hostBuf, size) == cudaSuccess)
                pinnedOutputBuffers[outName] = hostBuf;
        }
    }
}

size_t Detector::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0) return 0;
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}

size_t Detector::getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT8:  return 1;
    default: return 0;
    }
}

void Detector::loadEngine(const std::string& modelFile)
{
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();

    if (extension == ".engine")
    {
        std::cout << "[Detector] Loading engine: " << modelFile << std::endl;
        engine.reset(loadEngineFromFile(modelFile, runtime.get()));
    }
    else if (extension == ".onnx")
    {
        std::string engineFilePath = modelPath.replace_extension(".engine").string();
        if (!fileExists(engineFilePath))
        {
            std::cout << "[Detector] Building engine from ONNX: " << modelFile << std::endl;
            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile, gLogger);
            if (builtEngine)
            {
                nvinfer1::IHostMemory* serialized = builtEngine->serialize();
                if (serialized)
                {
                    std::ofstream ef(engineFilePath, std::ios::binary);
                    if (ef)
                    {
                        ef.write(reinterpret_cast<const char*>(serialized->data()),
                                 static_cast<std::streamsize>(serialized->size()));
                        ef.close();
                        config.ai_model = std::filesystem::path(engineFilePath).filename().string();
                        config.saveConfig("config.ini");
                        std::cout << "[Detector] Engine saved: " << engineFilePath << std::endl;
                    }
                    delete serialized;
                }
                delete builtEngine;
            }
        }
        engine.reset(loadEngineFromFile(engineFilePath, runtime.get()));
    }
    else
    {
        std::cerr << "[Detector] Unsupported model format: " << extension << std::endl;
    }
}

void Detector::processFrame(const cv::cuda::GpuMat& frame)
{
    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes.clear();
        detectedClasses.clear();
        return;
    }

    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame;
    frameReady   = true;
    inferenceCV.notify_one();
}

void Detector::inferenceThread()
{
    while (!shouldExit)
    {
        if (detector_model_changed.load())
        {
            destroyCudaGraph();
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                context.reset();
                engine.reset();
                for (auto& b : inputBindings)  if (b.second) cudaFree(b.second);
                for (auto& b : outputBindings) if (b.second) cudaFree(b.second);
                inputBindings.clear();
                outputBindings.clear();
            }
            initialize("models/" + config.ai_model);
            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
        }

        cv::cuda::GpuMat frame;
        bool hasNewFrame = false;

        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            if (!frameReady && !shouldExit)
                inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });
            if (shouldExit) break;
            if (frameReady)
            {
                frame = std::move(currentFrame);
                frame.download(currentFrameBGR);
                frameReady   = false;
                hasNewFrame  = true;
            }
        }

        if (!context)
        {
            if (!errorLogged) {
                std::cerr << "[Detector] Context not initialised" << std::endl;
                errorLogged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        errorLogged = false;

        if (hasNewFrame && !frame.empty())
        {
            try
            {
                preProcess(frame);

                if (useCudaGraph)
                {
                    if (!cudaGraphCaptured)
                        captureCudaGraph();
                    else
                        launchCudaGraph();
                }
                else
                {
                    context->enqueueV3(stream);
                }

                cudaStreamSynchronize(stream);

                for (const auto& name : outputNames)
                {
                    size_t size              = outputSizes[name];
                    nvinfer1::DataType dtype = outputTypes[name];

                    if (dtype == nvinfer1::DataType::kHALF)
                    {
                        size_t numElem = size / sizeof(__half);
                        auto&  halfBuf = outputDataBuffersHalf[name];
                        halfBuf.resize(numElem);
                        cudaMemcpy(halfBuf.data(), outputBindings[name], size,
                                   cudaMemcpyDeviceToHost);
                        std::vector<float> floatVec(numElem);
                        for (size_t i = 0; i < numElem; ++i)
                            floatVec[i] = __half2float(halfBuf[i]);
                        postProcess(floatVec.data(), name);
                    }
                    else if (dtype == nvinfer1::DataType::kFLOAT)
                    {
                        auto& floatBuf = outputDataBuffers[name];
                        floatBuf.resize(size / sizeof(float));
                        cudaMemcpy(floatBuf.data(), outputBindings[name], size,
                                   cudaMemcpyDeviceToHost);
                        postProcess(floatBuf.data(), name);
                    }
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Detector] Inference error: " << e.what() << std::endl;
            }
        }
    }
}

void Detector::releaseDetections()
{
    std::lock_guard<std::mutex> lock(detectionMutex);
    detectedBoxes.clear();
    detectedClasses.clear();
}

bool Detector::getLatestDetections(std::vector<cv::Rect>& boxes, std::vector<int>& classes)
{
    std::lock_guard<std::mutex> lock(detectionMutex);
    if (!detectedBoxes.empty()) {
        boxes   = detectedBoxes;
        classes = detectedClasses;
        return true;
    }
    return false;
}

// Resizes the input frame to 640×640, normalises to [0,1], splits channels,
// and writes the result into the TensorRT input binding in NCHW layout.
void Detector::preProcess(const cv::cuda::GpuMat& frame)
{
    if (frame.empty()) return;
    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) return;

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = static_cast<int>(dims.d[1]);
    int h = static_cast<int>(dims.d[2]);
    int w = static_cast<int>(dims.d[3]);

    try
    {
        cv::cuda::resize(frame, resizedBuffer, cv::Size(w, h), 0, 0,
                         cv::INTER_LINEAR, preprocessCvStream);
        resizedBuffer.convertTo(floatBuffer, CV_32F, 1.0f / 255.0f, 0,
                                preprocessCvStream);
        cv::cuda::split(floatBuffer, channelBuffers, preprocessCvStream);
        preprocessCvStream.waitForCompletion();

        size_t channelSize = static_cast<size_t>(h) * static_cast<size_t>(w) * sizeof(float);
        for (int i = 0; i < c; ++i)
        {
            cudaMemcpyAsync(
                static_cast<float*>(inputBuffer) + i * h * w,
                channelBuffers[i].ptr<float>(),
                channelSize,
                cudaMemcpyDeviceToDevice,
                stream
            );
        }
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[Detector] OpenCV error in preProcess: " << e.what() << std::endl;
    }
}

// Runs the appropriate YOLO postprocessor, applies per-class detect/color filters,
// and publishes results to detectedBoxes/detectedClasses.
void Detector::postProcess(const float* output, const std::string& outputName)
{
    if (numClasses <= 0) return;

    std::vector<Detection> detections;

    if (config.postprocess == "yolo10")
    {
        detections = postProcessYolo10(
            output,
            outputShapes[outputName],
            numClasses,
            config.confidence_threshold,
            config.nms_threshold
        );
    }
    else if (config.postprocess == "yolo8"  ||
             config.postprocess == "yolo9"  ||
             config.postprocess == "yolo11" ||
             config.postprocess == "yolo12")
    {
        auto shapeDims = context->getTensorShape(outputName.c_str());
        std::vector<int64_t> engineShape;
        for (int i = 0; i < shapeDims.nbDims; ++i)
            engineShape.push_back(shapeDims.d[i]);

        detections = postProcessYolo11(
            output,
            engineShape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold
        );
    }

    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes.clear();
        detectedClasses.clear();

        for (const auto& det : detections)
        {
            int classId = static_cast<int>(det.classId);
            if (!config.detect[classId]) continue;

            cv::Rect box = det.box & cv::Rect(0, 0,
                currentFrameBGR.cols, currentFrameBGR.rows);
            if (box.area() == 0) continue;

            // Head boxes nested inside a player box pass without color filtering
            if (classId == config.class_head)
            {
                bool nestedInside = false;
                for (size_t k = 0; k < detectedBoxes.size(); ++k)
                {
                    if (detectedClasses[k] == config.class_player &&
                        (detectedBoxes[k] & box) == box)
                    {
                        nestedInside = true;
                        break;
                    }
                }
                if (nestedInside) {
                    detectedBoxes.push_back(box);
                    detectedClasses.push_back(classId);
                    continue;
                }
            }

            // Skip color filter for classes with color[classId] == false
            if (!config.color[classId])
            {
                detectedBoxes.push_back(box);
                detectedClasses.push_back(classId);
                continue;
            }

            // HSV color filter: reject boxes without a sufficiently large color blob
            cv::Mat roiBGR = currentFrameBGR(box);
            cv::Mat hsv, mask;
            cv::cvtColor(roiBGR, hsv, cv::COLOR_BGR2HSV);
            cv::inRange(hsv,
                cv::Scalar(config.hsv_lower_h, config.hsv_lower_s, config.hsv_lower_v),
                cv::Scalar(config.hsv_upper_h, config.hsv_upper_s, config.hsv_upper_v),
                mask);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            bool hasValidContour = false;
            for (const auto& c : contours)
            {
                if (cv::contourArea(c) >= static_cast<double>(config.contour_area))
                {
                    hasValidContour = true;
                    break;
                }
            }

            if (hasValidContour)
            {
                detectedBoxes.push_back(box);
                detectedClasses.push_back(classId);
            }
        }

        detectionVersion++;
    }

    detectionCV.notify_one();
}
