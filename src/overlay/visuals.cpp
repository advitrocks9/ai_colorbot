#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "overlay/visuals.h"
#include "overlay/overlay.h"
#include "app.h"
#include "capture/capture.h"

void displayThread()
{
    if (config.show_window)
    {
        cv::namedWindow("Debug", cv::WINDOW_NORMAL);
        cv::setWindowProperty("Debug", cv::WND_PROP_TOPMOST, config.always_on_top ? 1 : 0);
    }

    int currentSize = 0;

    while (!shouldExit)
    {
        if (show_window_changed.load())
        {
            if (config.show_window)
            {
                cv::namedWindow("Debug", cv::WINDOW_NORMAL);
                cv::setWindowProperty("Debug", cv::WND_PROP_TOPMOST, config.always_on_top ? 1 : 0);
            }
            else
            {
                cv::destroyWindow("Debug");
            }
            show_window_changed.store(false);
        }

        if (config.show_window)
        {
            cv::Mat frame;

            if (config.capture_use_cuda)
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                while (latestFrameGpu.empty() && !shouldExit)
                {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    lock.lock();
                }
                if (shouldExit) break;
                latestFrameGpu.download(frame);
            }
            else
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCV.wait(lock, [] { return !latestFrameCpu.empty() || shouldExit.load(); });
                if (shouldExit) break;
                frame = latestFrameCpu.clone();
            }

            if (frame.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            cv::Mat displayFrame;
            if (config.window_size != 100)
            {
                int desiredSize = static_cast<int>((640 * config.window_size) / 100);
                if (desiredSize != currentSize)
                {
                    cv::resizeWindow("Debug", desiredSize, desiredSize);
                    currentSize = desiredSize;
                }
                cv::resize(frame, displayFrame, cv::Size(desiredSize, desiredSize));
            }
            else
            {
                displayFrame = frame.clone();
            }

            std::vector<cv::Rect> boxes;
            std::vector<int> classes;
            if (detector.getLatestDetections(boxes, classes))
            {
                float boxScaleX = static_cast<float>(displayFrame.cols) / config.detection_resolution;
                float boxScaleY = static_cast<float>(displayFrame.rows) / config.detection_resolution;

                for (size_t i = 0; i < boxes.size(); ++i)
                {
                    cv::Rect box = boxes[i];
                    box.x      = static_cast<int>(box.x      * boxScaleX);
                    box.y      = static_cast<int>(box.y      * boxScaleY);
                    box.width  = static_cast<int>(box.width  * boxScaleX);
                    box.height = static_cast<int>(box.height * boxScaleY);

                    cv::rectangle(displayFrame, box, cv::Scalar(0, 255, 0), 2);
                    cv::putText(displayFrame,
                        std::to_string(classes[i]),
                        cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 0),
                        1);
                }
            }

            if (config.show_fps)
            {
                cv::putText(displayFrame,
                    "FPS: " + std::to_string(static_cast<int>(captureFps)),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(255, 255, 0),
                    2);
            }

            cv::imshow("Debug", displayFrame);
            if (cv::waitKey(1) == 27)
                shouldExit = true;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
}
