// TensorRT inference engine wrapper.
// Runs inference on a background thread, publishes detections via version counter.
#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <NvInfer.h>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unordered_map>
#include <cuda_fp16.h>
#include <memory>
#include <thread>
#include <chrono>
#include <functional>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

class Detector
{
public:
    Detector();
    ~Detector();

    void initialize(const std::string& modelFile);
    void processFrame(const cv::cuda::GpuMat& frame);
    void inferenceThread();

    void releaseDetections();
    bool getLatestDetections(std::vector<cv::Rect>& boxes, std::vector<int>& classes);

    std::mutex detectionMutex;
    int detectionVersion = 0;
    std::condition_variable detectionCV;
    std::vector<cv::Rect> detectedBoxes;
    std::vector<int>      detectedClasses;

    float img_scale = 1.0f;

    int getInputHeight() const { return static_cast<int>(inputDims.d[2]); }
    int getInputWidth()  const { return static_cast<int>(inputDims.d[3]); }

    std::vector<std::string>                   inputNames;
    std::vector<std::string>                   outputNames;
    std::unordered_map<std::string, size_t>    outputSizes;

private:
    std::unique_ptr<nvinfer1::IRuntime>          runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    nvinfer1::Dims inputDims;

    cv::cuda::Stream cvStream;
    cv::cuda::Stream preprocessCvStream;
    cv::cuda::Stream postprocessCvStream;

    cudaStream_t stream           = nullptr;
    cudaStream_t preprocessStream = nullptr;
    cudaStream_t postprocessStream = nullptr;

    // Optional CUDA graph for repeated inference on fixed-size inputs
    bool              useCudaGraph      = false;
    bool              cudaGraphCaptured = false;
    cudaGraph_t       cudaGraph         = nullptr;
    cudaGraphExec_t   cudaGraphExec     = nullptr;
    void captureCudaGraph();
    void launchCudaGraph();
    void destroyCudaGraph();

    // Page-locked host buffers for faster DtoH memcpy
    std::unordered_map<std::string, void*> pinnedOutputBuffers;
    bool usePinnedMemory = false;

    std::mutex inferenceMutex;
    std::condition_variable inferenceCV;
    std::atomic<bool> shouldExit{false};
    cv::cuda::GpuMat currentFrame;
    cv::Mat          currentFrameBGR;
    bool             frameReady = false;

    void loadEngine(const std::string& engineFile);
    void preProcess(const cv::cuda::GpuMat& frame);
    void postProcess(const float* output, const std::string& outputName);
    void getInputNames();
    void getOutputNames();
    void getBindings();

    std::unordered_map<std::string, size_t>               inputSizes;
    std::unordered_map<std::string, void*>                inputBindings;
    std::unordered_map<std::string, void*>                outputBindings;
    std::unordered_map<std::string, std::vector<int64_t>> outputShapes;
    int numClasses = 0;

    size_t getSizeByDim(const nvinfer1::Dims& dims);
    size_t getElementSize(nvinfer1::DataType dtype);

    std::string  inputName;
    void*        inputBufferDevice = nullptr;
    std::unordered_map<std::string, std::vector<float>>  outputDataBuffers;
    std::unordered_map<std::string, std::vector<__half>> outputDataBuffersHalf;
    std::unordered_map<std::string, nvinfer1::DataType>  outputTypes;

    cv::cuda::GpuMat resizedBuffer;
    cv::cuda::GpuMat floatBuffer;
    std::vector<cv::cuda::GpuMat> channelBuffers;
};

#endif // DETECTOR_H
