// Frame acquisition interface and CaptureThread wrapper.
#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <thread>

class CaptureThread {
public:
    CaptureThread(int resolution, bool useCuda, double fps, int monitorIdx);
    ~CaptureThread();
    void updateConfig(int resolution, bool useCuda, double fps, int monitorIdx);

private:
    void runLoop();

    int    resolution_;
    bool   useCuda_;
    double fps_;
    int    monitor_idx_;
    std::thread thread_;
};

void captureThread(int captureWidth, int captureHeight);

extern int screenWidth;
extern int screenHeight;

extern std::atomic<int> captureFrameCount;
extern std::atomic<int> captureFps;
extern std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

extern cv::cuda::GpuMat latestFrameGpu;
extern cv::Mat          latestFrameCpu;

extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;

class IScreenCapture {
public:
    virtual ~IScreenCapture() {}
    virtual cv::cuda::GpuMat GetNextFrameGpu() = 0;
    virtual cv::Mat          GetNextFrameCpu() = 0;
};

#endif // CAPTURE_H
