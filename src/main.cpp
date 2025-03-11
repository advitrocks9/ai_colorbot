// Main entry point. Initialises subsystems and spawns pipeline threads.
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include "capture/capture.h"
#include "overlay/visuals.h"
#include "detection/detector.h"
#include "input/mouse.h"
#include "overlay/overlay.h"
#include "app.h"
#include "util/otherTools.h"
#include "input/makcuConnection.h"

std::atomic<bool> shouldExit(false);
std::mutex configMutex;

Detector detector;
MouseThread* globalMouseThread = nullptr;
CaptureThread* globalCaptureThread = nullptr;
Config config;
MakcuConnection* makcuSerial = nullptr;

std::atomic<bool> shooting(false);
std::atomic<bool> netAimbot(false);
std::atomic<bool> netTrigger(false);

void initializeInputMethod()
{
    std::lock_guard<std::mutex> lock(globalMouseThread->input_method_mutex);
    if (makcuSerial) {
        delete makcuSerial;
        makcuSerial = nullptr;
    }
    makcuSerial = new MakcuConnection();
    if (!makcuSerial->isOpen()) {
        std::cerr << "[Makcu] Error connecting to Makcu." << std::endl;
        delete makcuSerial;
        makcuSerial = nullptr;
    }
    globalMouseThread->setMakcuConnection(makcuSerial);
}

// Listens on UDP port 5005 for remote tracking/auto-click toggle commands.
// Message format: "true,false" (tracking,auto-click as booleans).
void udpListenerThread()
{
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in serverAddr = {};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(5005);
    bind(sock, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr));
    char buf[64];
    while (!shouldExit.load())
    {
        sockaddr_in sender = {};
        int senderSize = sizeof(sender);
        int recvLen = recvfrom(sock, buf, sizeof(buf) - 1, 0,
                               reinterpret_cast<sockaddr*>(&sender), &senderSize);
        if (recvLen > 0)
        {
            buf[recvLen] = '\0';
            std::string msg(buf, static_cast<size_t>(recvLen));
            auto commaPos = msg.find(',');
            if (commaPos != std::string::npos)
            {
                bool newA = (msg.substr(0, commaPos) == "true");
                bool newT = (msg.substr(commaPos + 1) == "true");
                if (newA != netAimbot.load())
                {
                    netAimbot.store(newA);
                    std::cout << "Tracking " << (newA ? "enabled" : "disabled") << std::endl;
                }
                if (newT != netTrigger.load())
                {
                    netTrigger.store(newT);
                    std::cout << "Auto-click " << (newT ? "enabled" : "disabled") << std::endl;
                }
            }
        }
    }
    closesocket(sock);
    WSACleanup();
}

// Consumes detection results, drives PID mouse movement, and triggers
// auto-click when the nearest target is within the center deadzone.
void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastDetectionVersion = -1;
    int noTargetFrames = 0;
    const int NO_TARGET_RESET_FRAMES = 10;
    std::chrono::milliseconds timeout(30);

    using Clock = std::chrono::steady_clock;
    auto lastShotTime = Clock::now() - std::chrono::milliseconds(300);
    const double centerDeadzone = 5.0;
    const std::chrono::duration<double> cooldown(0.3);

    while (!shouldExit.load())
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> classes;

        {
            std::unique_lock<std::mutex> lock(detector.detectionMutex);
            detector.detectionCV.wait_for(lock, timeout, [&]() {
                return detector.detectionVersion > lastDetectionVersion || shouldExit.load();
            });
            if (shouldExit.load()) break;
            if (detector.detectionVersion <= lastDetectionVersion) continue;
            lastDetectionVersion = detector.detectionVersion;
            boxes   = detector.detectedBoxes;
            classes = detector.detectedClasses;
        }

        if (detection_resolution_changed.load())
        {
            std::lock_guard<std::mutex> cfgLock(configMutex);
            globalCaptureThread->updateConfig(
                config.detection_resolution,
                config.capture_use_cuda,
                config.capture_fps,
                config.monitor_idx
            );
            mouseThread.updateConfig(
                config.detection_resolution,
                config.dpi,
                config.sensitivity,
                config.fovX,
                config.fovY,
                config.pid_kp,
                config.pid_ki,
                config.pid_kd,
                config.capture_fps
            );
            detection_resolution_changed.store(false);
        }

        AimbotTarget* target = sortTargets(
            boxes,
            classes,
            config.detection_resolution,
            config.detection_resolution,
            config.disable_headshot
        );

        bool doAim    = netAimbot.load();
        bool doTrigger = netTrigger.load();

        if (target)
        {
            if (doAim)
                mouseThread.moveMousePivot(target->pivotX, target->pivotY);
            if (doTrigger) {
                double dx = target->pivotX - (config.detection_resolution / 2.0);
                double dy = target->pivotY - (config.detection_resolution / 2.0);
                if (std::abs(dx) <= centerDeadzone && std::abs(dy) <= centerDeadzone) {
                    auto now = Clock::now();
                    if (now - lastShotTime >= cooldown) {
                        mouseThread.pressMouse(*target);
                        mouseThread.releaseMouse();
                        lastShotTime = now;
                    }
                }
            }
        }
        else
        {
            noTargetFrames++;
            if (noTargetFrames > NO_TARGET_RESET_FRAMES)
            {
                mouseThread.resetPID();
                noTargetFrames = 0;
            }
            if (doTrigger)
                mouseThread.releaseMouse();
        }

        delete target;
    }
}

int main()
{
    try
    {
        int cudaDevices = 0;
        if (cudaGetDeviceCount(&cudaDevices) != cudaSuccess || cudaDevices == 0)
        {
            std::cerr << "[MAIN] CUDA required but no devices found." << std::endl;
            std::cin.get();
            return -1;
        }

        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

        if (!config.loadConfig())
        {
            std::cerr << "[Config] Error loading config!" << std::endl;
            std::cin.get();
            return -1;
        }

        MouseThread mouseThread(
            config.detection_resolution,
            config.dpi,
            config.sensitivity,
            config.fovX,
            config.fovY,
            config.pid_kp,
            config.pid_ki,
            config.pid_kd,
            config.capture_fps
        );
        globalMouseThread = &mouseThread;

        CaptureThread captureObj(
            config.detection_resolution,
            config.capture_use_cuda,
            config.capture_fps,
            config.monitor_idx
        );
        globalCaptureThread = &captureObj;

        auto availableModels = getAvailableModels();
        if (availableModels.empty())
        {
            std::cerr << "[MAIN] No AI models found in models/." << std::endl;
            std::cin.get();
            return -1;
        }
        if (!std::filesystem::exists("models/" + config.ai_model))
        {
            config.ai_model = availableModels[0];
            config.saveConfig();
        }

        detector.initialize("models/" + config.ai_model);
        detection_resolution_changed.store(true);
        initializeInputMethod();

        std::thread udpThread(udpListenerThread);
        udpThread.detach();

        std::thread detThread(&Detector::inferenceThread, &detector);
        std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
        std::thread overlayThd(overlayThread);

        welcomeMessage();
        displayThread();

        detThread.join();
        mouseMovThread.join();
        overlayThd.join();

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[MAIN] Fatal error: " << e.what() << std::endl;
        std::cin.get();
        return -1;
    }
}
