// Capture module — main capture loop and CaptureThread implementation.
// Drives DXGI Desktop Duplication at the configured FPS, applies circle masking
// if enabled, and feeds frames directly into the detector pipeline.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <timeapi.h>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include "capture/capture.h"
#include "overlay/overlay.h"
#include "detection/detector.h"
#include "app.h"
#include "util/otherTools.h"
#include "capture/duplicationApiCapture.h"

cv::cuda::GpuMat latestFrameGpu;
cv::Mat          latestFrameCpu;
std::mutex       frameMutex;

int screenWidth  = 0;
int screenHeight = 0;

std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

void captureThread(int captureWidth, int captureHeight)
{
    try
    {
        if (config.verbose)
        {
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;
            std::cout << "[Capture] CUDA devices found: "
                      << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
        }

        IScreenCapture* capturer = new DuplicationAPIScreenCapture(
            config.monitor_idx, captureWidth, captureHeight);

        if (config.verbose)
            std::cout << "[Capture] Using Duplication API on monitor "
                      << config.monitor_idx << std::endl;

        bool frameLimitingEnabled = false;
        std::chrono::duration<double, std::milli> frameDuration(0);

        if (config.capture_fps > 0.0)
        {
            timeBeginPeriod(1);
            frameDuration = std::chrono::duration<double, std::milli>(
                1000.0 / config.capture_fps);
            frameLimitingEnabled = true;
        }

        captureFpsStartTime = std::chrono::high_resolution_clock::now();

        cv::cuda::GpuMat circleMaskGpu, workGpu;
        auto startTime = std::chrono::high_resolution_clock::now();

        while (!shouldExit)
        {
            // React to FPS changes requested by the overlay
            if (capture_fps_changed.load())
            {
                if (config.capture_fps > 0.0)
                {
                    if (!frameLimitingEnabled) {
                        timeBeginPeriod(1);
                        frameLimitingEnabled = true;
                    }
                    frameDuration = std::chrono::duration<double, std::milli>(
                        1000.0 / config.capture_fps);
                }
                else
                {
                    if (frameLimitingEnabled) {
                        timeEndPeriod(1);
                        frameLimitingEnabled = false;
                    }
                    frameDuration = std::chrono::duration<double, std::milli>(0);
                }
                capture_fps_changed.store(false);
            }

            // Reinitialise capture when resolution changes
            if (detection_resolution_changed.load())
            {
                delete capturer;
                int newSize = config.detection_resolution;
                capturer = new DuplicationAPIScreenCapture(
                    config.monitor_idx, newSize, newSize);
                if (config.verbose)
                    std::cout << "[Capture] Re-init duplication API" << std::endl;
                detection_resolution_changed.store(false);
            }

            if (config.capture_use_cuda)
            {
                cv::cuda::GpuMat screenshotGpu = capturer->GetNextFrameGpu();
                if (screenshotGpu.empty())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                if (config.circle_mask)
                {
                    // Lazily build the circle mask on first use or on resolution change
                    if (circleMaskGpu.empty() || circleMaskGpu.size() != screenshotGpu.size())
                    {
                        cv::Mat maskHost(screenshotGpu.size(), CV_8UC1, cv::Scalar::all(0));
                        cv::circle(maskHost,
                                   { maskHost.cols / 2, maskHost.rows / 2 },
                                   std::min(maskHost.cols, maskHost.rows) / 2,
                                   cv::Scalar::all(255), -1);
                        circleMaskGpu.upload(maskHost);
                    }
                    screenshotGpu.copyTo(workGpu, circleMaskGpu);
                }

                cv::cuda::resize(
                    config.circle_mask ? workGpu : screenshotGpu,
                    workGpu,
                    cv::Size(config.detection_resolution, config.detection_resolution)
                );

                {
                    std::lock_guard<std::mutex> lk(frameMutex);
                    workGpu.swap(latestFrameGpu);
                    latestFrameCpu.release();
                }

                detector.processFrame(workGpu);
            }
            else
            {
                cv::Mat screenshotCpu = capturer->GetNextFrameCpu();
                if (screenshotCpu.empty())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                if (config.circle_mask)
                {
                    cv::Mat mask = cv::Mat::zeros(screenshotCpu.size(), CV_8UC1);
                    cv::Point center(mask.cols / 2, mask.rows / 2);
                    int radius = std::min(mask.cols, mask.rows) / 2;
                    cv::circle(mask, center, radius, cv::Scalar(255), -1);
                    screenshotCpu.copyTo(screenshotCpu, mask);
                }

                cv::resize(screenshotCpu, screenshotCpu,
                           cv::Size(captureWidth, captureHeight),
                           0, 0, cv::INTER_LINEAR);

                {
                    std::lock_guard<std::mutex> lk(frameMutex);
                    latestFrameCpu = screenshotCpu.clone();
                    latestFrameGpu.release();
                }
                frameCV.notify_one();

                cv::cuda::GpuMat toTrt;
                toTrt.upload(screenshotCpu);
                detector.processFrame(toTrt);
            }

            ++captureFrameCount;
            auto now = std::chrono::high_resolution_clock::now();
            if ((now - captureFpsStartTime).count() / 1e9 >= 1.0)
            {
                captureFps   = static_cast<int>(captureFrameCount.load());
                captureFrameCount = 0;
                captureFpsStartTime = now;
            }

            if (frameLimitingEnabled)
            {
                auto endTime      = std::chrono::high_resolution_clock::now();
                auto workDuration = endTime - startTime;
                auto sleepDuration = frameDuration - workDuration;
                if (sleepDuration > std::chrono::duration<double, std::milli>(0))
                    std::this_thread::sleep_for(sleepDuration);
                startTime = std::chrono::high_resolution_clock::now();
            }
        }

        if (frameLimitingEnabled)
            timeEndPeriod(1);

        delete capturer;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception: " << e.what() << std::endl;
    }
}

CaptureThread::CaptureThread(int resolution, bool useCuda, double fps, int monitorIdx)
    : resolution_(resolution)
    , useCuda_(useCuda)
    , fps_(fps)
    , monitor_idx_(monitorIdx)
{
    config.detection_resolution = resolution_;
    config.capture_use_cuda     = useCuda_;
    config.capture_fps          = static_cast<int>(fps_);
    config.monitor_idx          = monitorIdx;
    thread_ = std::thread(&CaptureThread::runLoop, this);
}

CaptureThread::~CaptureThread()
{
    shouldExit = true;
    if (thread_.joinable())
        thread_.join();
}

void CaptureThread::updateConfig(int resolution, bool useCuda, double fps, int monitorIdx)
{
    resolution_  = resolution;
    useCuda_     = useCuda;
    fps_         = fps;
    monitor_idx_ = monitorIdx;

    config.detection_resolution = resolution_;
    config.capture_use_cuda     = useCuda_;
    config.capture_fps          = static_cast<int>(fps_);
    config.monitor_idx          = monitorIdx;

    detection_resolution_changed.store(true);
    capture_fps_changed.store(true);
}

void CaptureThread::runLoop()
{
    captureThread(resolution_, resolution_);
}
