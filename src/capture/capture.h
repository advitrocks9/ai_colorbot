// Capture module — frame acquisition interface and CaptureThread wrapper.
// IScreenCapture defines the polymorphic capture interface; CaptureThread
// manages the background acquisition loop and responds to config changes.
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

// Shared change-notification flags written by the overlay and consumed by the
// capture/detection pipeline threads to trigger config reloads without restarts.
extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_fps_changed;

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

extern CaptureThread* globalCaptureThread;

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
extern std::atomic<bool> show_window_changed;

// Abstract interface for screen capture backends (currently DXGI Desktop Duplication).
class IScreenCapture {
public:
    virtual ~IScreenCapture() {}
    virtual cv::cuda::GpuMat GetNextFrameGpu() = 0;
    virtual cv::Mat          GetNextFrameCpu() = 0;
};

#endif // CAPTURE_H
